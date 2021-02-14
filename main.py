from flask import Flask,send_file,send_from_directory
from flask_restful import Api,Resource,reqparse
import sounddevice as sd
import librosa
import uuid
import json
import re
import os
import subprocess
import signal
import pathlib
import asyncio

BASE_DIR = str(pathlib.Path().absolute())
# checkpoint_pattern = r'^model_checkpoint_path: "(.*)"$'
print("BASE_DIR:",BASE_DIR)
STEP = 777
CHECKPOINT_PATH = os.path.join('logs-tacotron','model.ckpt-'+str(STEP))

# from_default = False
# if from_default:
#     with open(r'logs-tacotron\checkpoint') as checkpoint_fh:
#         match_obj = re.search(checkpoint_pattern, checkpoint_fh.readline())
#         if match_obj is not None:
#             CHECKPOINT_PATH = match_obj.group(1).replace(r"\\", "\\").replace("/", "\\")
#             print("Found checkpoint path...")
print(CHECKPOINT_PATH)
from synthesizer import Synthesizer
syn = Synthesizer()
syn.load(checkpoint_path=CHECKPOINT_PATH)

pid = None
pre_pid = None

app = Flask(__name__)
app.config["AUDIO_DIR"] = os.path.join(BASE_DIR, "flask-gen")
api = Api(app)
audio_args=reqparse.RequestParser()
train_args=reqparse.RequestParser()
train_args.add_argument("base_dir",help="The name of audio file",type=str)
play_audio_args=reqparse.RequestParser()
audio_args.add_argument("text",help="Request does not contain text to generate audio",type=str,default="This is a test audio from  text to speech")
audio_args.add_argument("file_name",default="AudioFile",help="The name of audio file",type=str)
play_audio_args.add_argument("text",help="Request does not contain text to generate audio",required=True)
play_audio_args.add_argument("audio_device",help="On which audio device you want to play the audio",default="default")

req_tts_args = reqparse.RequestParser()
req_tts_args.add_argument("text", help="Input text", type=str)

req_audio_args = reqparse.RequestParser()
req_audio_args.add_argument("file_name", help="Audio file name", type=str)

pre_process_args = reqparse.RequestParser()
pre_process_args.add_argument("base_dir",help="You must provide dataset path", default=BASE_DIR)
pre_process_args.add_argument("dataset",help="You must provide dataset type", default="ljspeech")

def save_wav(audio_binary, file_name="sample.wav"):
    with open(file_name, "wb") as wavfile:
        wavfile.write(audio_binary)


class TextToSpeech(Resource):
    #Function to generate audio with given text
    def post(self):
        args=audio_args.parse_args()
        save_wav(syn.synthesize(args["text"]), os.path.join(app.config["AUDIO_DIR"],args["file_name"]))
        return send_from_directory(app.config["AUDIO_DIR"],filename=args["file_name"]+".wav",as_attachment=True)

    def get(self, text):
        # args=audio_args.parse_args()
        print(text)
        fileName = "".join(x for x in text if x.isalnum())
        save_wav(syn.synthesize(text), os.path.join(app.config["AUDIO_DIR"],fileName+".wav"))
        print("Audio",os.path.join(app.config["AUDIO_DIR"],fileName+".wav"))
        return send_from_directory(app.config["AUDIO_DIR"],filename=fileName+".wav",as_attachment=True)                                                                                                                                                                                                                                                                 

class PlayAudio(Resource):
    def post(self):
        args=play_audio_args.parse_args()
        save_wav(syn.synthesize(args["text"]), "playaudio.wav")
        sd.default.device=args['audio_device']
        sd.default.samplerate=48000
        audio,_=librosa.load("playaudio.wav",48000)
        sd.play(audio)

class TrainStatus(Resource):
    def get(self):
        global pid
        config_fh = open("config.json", "r")
        status_fh = open("status.json", "r")
        config = json.loads(config_fh.read())
        status = json.loads(status_fh.read())
        config_fh.close()
        status_fh.close()
        return {
            "isTraining": config['isTraining'],
            "UUID": config['UUID'],
            "step": status['step'],
            "avg_loss": status['avg_loss'],
            "loss": status['loss'],
            "restore_step": status['restore_step'],
            "pending": (not config['isTraining']) and (pid is not None)
        }

class TrainModel(Resource):
    def trainConfig(self):
        global pid, BASE_DIR, pre_pid
        pre_status = None
        config = None
        restore_step = None
        with open("status.json") as fh:
            restore_step = json.loads(fh.read())['restore_step']

        with open("preprocess.json", "r") as fh:
            pre_status = json.loads(fh.read())

        if (pre_pid is not None) or pre_status['status']:
            return trainRes("preprocess_not_completed")

        with open("config.json", "r") as fh:
            config = json.loads(fh.read())
        if config['isTraining']:
            return trainRes("already_running", config['UUID'])
        else:
            if pid is not None:
                return trainRes("already_running", config['UUID'])
            cmd = "python train.py --restore_step=%s --base_dir %s" % (restore_step, BASE_DIR) # Line here
            if restore_step == 0:
                cmd = "python train.py --base_dir %s" % (BASE_DIR) # Line here
            pid = subprocess.Popen(cmd,shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            return trainRes("training_request_sent")

    def get(self):
        return self.trainConfig()


def trainRes(status, UUID = ""):
    ''' Returns the train start/stop response
    '''
    return {"status": status, "UUID": UUID}


class TrainStop(Resource):
    def get(self):
        global pid
        UUID = ""
        with open("config.json", "r") as fh:
            config = json.loads(fh.read())
            UUID = config['UUID']
        
        with open("config.json", "w") as fh:
            fh.write('{"isTraining": false, "UUID": ""}')

        if pid is None:
            return trainRes("training_not_started")

        pid.send_signal(signal.CTRL_BREAK_EVENT)
        pid.kill()
        pid = None
        return trainRes("training_stopped", UUID)

class PreProcessModel(Resource):
    def post(self):
        args=pre_process_args.parse_args()
        print(args['dataset'])
        global pre_pid
        if pre_pid is not None:
            pre_pid.send_signal(signal.CTRL_BREAK_EVENT)
            pre_pid.kill()
            pre_pid = None
        cmd = "python preprocess.py --dataset "+args["dataset"]  +" --base_dir %s" % args["base_dir"] # Line here
        pre_pid = subprocess.Popen(cmd,shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        return trainRes("preprocess_request_sent")

    def get(self):
        args=pre_process_args.parse_args()
        global pre_pid
        if pre_pid is not None:
            pre_pid.send_signal(signal.CTRL_BREAK_EVENT)
            pre_pid.kill()
            pre_pid = None
        cmd = "python preprocess.py --dataset "+args["dataset"]  +" --base_dir %s" % args["base_dir"] # Line here
        pre_pid = subprocess.Popen(cmd,shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        return trainRes("preprocess_request_sent")
        
class PreProcessStop(Resource):

    def get(self):
        global pre_pid
        with open("preprocess.json", "w") as fh:
            fh.write('{"status": false}')
        
        if pre_pid is not None:
            pre_pid.send_signal(signal.CTRL_BREAK_EVENT)
            pre_pid.kill()
            pre_pid = None
        return trainRes("preprocess_stopped")

def setCheckpoint(step):
    global syn, CHECKPOINT_PATH, STEP
    if step is None:
        return {"checkpoint": str(STEP)}
    if os.path.exists(os.path.join(BASE_DIR, 'logs-tacotron', 'model.ckpt-'+str(step)+'.index')):
        try:
            syn = Synthesizer()
            CHECKPOINT_PATH = os.path.join('logs-tacotron', 'model.ckpt-'+str(step))
            syn.load(checkpoint_path=CHECKPOINT_PATH)
            STEP = step
            return { "checkpoint": str(step), "status":"request_success"}
        except Exception as e:
            print(e)
            return {"status": "error"}
    
    try:
        step = 777
        syn = Synthesizer()
        CHECKPOINT_PATH = os.path.join('logs-tacotron', 'model.ckpt-'+str(step))
        syn.load(checkpoint_path=CHECKPOINT_PATH)
        STEP = step
        return { "checkpoint": str(STEP), "status":"request_alternate_success"}
    except Exception as e:
        print(e)
        return {"status":"error"}


def getCheckpoint():
    global STEP
    return {"checkpoint": str(STEP)}

class TrainCheckPointGet(Resource):
    def get(self):
        return getCheckpoint()

class TrainCheckPointSet(Resource):
    def get(self, step):
        return setCheckpoint(step)

api.add_resource(TextToSpeech, "/audio/file/<string:text>")
api.add_resource(PlayAudio, "/output/audio")
api.add_resource(TrainModel, "/train")
api.add_resource(TrainStop, '/train/stop')
api.add_resource(TrainStatus, "/train/status")
api.add_resource(PreProcessModel, "/preprocess")
api.add_resource(PreProcessStop, '/preprocess/stop')
api.add_resource(TrainCheckPointSet, '/checkpoint/<int:step>')
api.add_resource(TrainCheckPointGet, '/checkpoint')

print("Apis Added")
print(app.config["AUDIO_DIR"])
from flask_cors import CORS
CORS(app)
if __name__=="__main__":
    app.run(debug=True,port=7777, host='0.0.0.0')