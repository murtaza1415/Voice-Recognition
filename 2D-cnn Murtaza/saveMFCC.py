import os
import random
import numpy as np
from glob import glob
import librosa
import gc



gc.disable()


#Load the data
audio_rate = 16000
n_mfcc = 60
no_seconds = 4
speakers = glob("data/evaluation_normalized/*/")

x_and_y = []
x_train = []
y_train = []
no_speakers_loaded = 0

for speaker in speakers:
    label = os.path.split(os.path.split(speaker)[0])[1]                  #Read speaker ID from path.
    audio_files = glob(speaker + '*.wav')
    for audio_file in audio_files:
        audio_data, _ = librosa.load(sr=audio_rate, mono=True, path=audio_file)
        assert _ == audio_rate
        
        no_clips = len(audio_data) / (audio_rate*no_seconds)
        #print(no_clips)
        for i in range(0, no_clips):                                                                #4 sec clips
            audio_clip = audio_data[(audio_rate*no_seconds)*i:(audio_rate*no_seconds)*(i+1)]
            audio_mfcc = librosa.feature.mfcc(y=audio_clip, sr=audio_rate,  n_mfcc=n_mfcc)
            audio_mfcc = np.reshape(audio_mfcc, (-1,audio_mfcc.shape[1],1))
            x_and_y.append([audio_mfcc, label])                                        
    no_speakers_loaded += 1
    print(no_speakers_loaded)
    
gc.enable()
    
#Correctly reorganize arrays
random.shuffle(x_and_y)        
x_and_y = np.asarray(x_and_y)
x_train = x_and_y[:,0]
x_train = np.stack(x_train)
y_train = x_and_y[:,1]
y_train = np.reshape(y_train, (-1,1))

#Save the numpy data
np.save('data/x_eval.npy', x_train)
np.save('data/y_eval.npy', y_train)