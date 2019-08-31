import os
import shutil
from glob import glob
import librosa

#Delete audios less than 4 sec.
no_audios = 0
small_audios = []
audio_files = glob("data/LibriSpeech/train-clean-100/*/*.flac")
for audio_file in audio_files:
    audio_data, audio_rate = librosa.load(sr=500, mono=True, path=audio_file)
    no_audios += 1
    if (no_audios % 100 == 0):
        print(no_audios)
    if len(audio_data) < (audio_rate*4):
        small_audios.append(audio_file)
print('\n-----------------------------------------------------------------------')
print(len(small_audios))
for audio_file in small_audios:
    os.remove(audio_file)