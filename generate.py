import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--steps', type=int, default=63500)
parser.add_argument('-t', '--text', type=str)
parser.add_argument('-o', '--output', type=str, default='sample.wav')
args = parser.parse_args()

CHECKPOINT_PATH = r'logs-tacotron\model.ckpt-'+str(args.steps)

from synthesizer import Synthesizer

syn = Synthesizer()
syn.load(checkpoint_path=CHECKPOINT_PATH)

def save_wav(audio_binary, file_name="sample1.wav"):
    with open(file_name, "wb") as wavfile:
        wavfile.write(audio_binary)
        
save_wav(syn.synthesize(args.text), args.output)
print("Wav file is created at "+args.output)