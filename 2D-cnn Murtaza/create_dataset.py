import os
import shutil
import random
import librosa
import numpy as np
from glob import glob
from scipy import signal
from IPython import display
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization
from python_speech_features import mfcc

#For fft spectrum.
import sigproc
import constants as c
from scipy.signal import lfilter, butter

import psutil
p = psutil.Process()
p.cpu_affinity([0,1,2,5,9,13,17,18,19,20,23,27,28,29,30,31])




#Functions for fft spectrum. Copied from 'https://github.com/linhdvu14/vggvox-speaker-identification/blob/master'
def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


def get_fft_spectrum(signal, buckets=None):
	#signal = load_wav(filename,c.SAMPLE_RATE)
	signal *= 2**15

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)

	# truncate to max bucket sizes
	#rsize = max(k for k in buckets if k <= fft_norm.shape[1])
	#rstart = int((fft_norm.shape[1]-rsize)/2)
	#out = fft_norm[:,rstart:rstart+rsize]

	#return out
	return fft_norm





'''Note: Study required for optimal spectogram.'''

def audio_to_image(audio, sr):                            
    '''
    Convert an audio to its spectrogram.
    '''
    #_,_,spectrogram = signal.spectrogram(audio, sr)
    #audio_mfcc = librosa.feature.mfcc(y=audio, sr=16000,  n_mfcc=60)
    #audio_mfcc2 = mfcc(signal=audio, samplerate=16000, nfft=512, winlen=0.025, winstep=0.009, nfilt=400, numcep = 300, winfunc=np.hamming)
    audio_mfcc3 = get_fft_spectrum(audio)
    return audio_mfcc3





def get_label(path):
    '''
    Extract label of audio from the given path.
    '''
    label = os.path.split(os.path.split(path)[0])[1]
    label = int(label)
    return label





def create_training_dataset_new():
    '''
    Create the complete dataset for training and the associated labels. 
    The dataset is created and saved in subsets due to limitaion of RAM.
    Note that the audios are converted to images.
    '''
    
    audio_files = glob("/data/techresearch/Murtaza/vox1/dev/wav/*/*.wav")
    random.shuffle(audio_files)
    
    subset_size = 3000
    total_subsets = 49
    audio_rate = 16000
    no_seconds = 3
    
    for j in range(total_subsets):
        x_train = []
        y_train = []
        
        audio_subset = audio_files[ (subset_size*j) : (subset_size*(j+1)) ]
        print(str((subset_size*j)) + ' to ' + str(subset_size*(j+1)))
        
        for audio_file in audio_subset:
            label = get_label(audio_file)
            audio_data, _ = librosa.load(sr=audio_rate, mono=True, path=audio_file)

            assert _ == audio_rate
            no_clips = len(audio_data) // (audio_rate*no_seconds)

            for i in range(0, no_clips):                                                                #4 sec clips
                audio_clip = audio_data[(audio_rate*no_seconds)*i:(audio_rate*no_seconds)*(i+1)]
                image = audio_to_image(audio_clip, audio_rate)
                x_train.append(image)
                y_train.append(label)
    
        #Convert to numpy
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        #Save the numpy data
        np.save('/data/techresearch/Murtaza/mfcc3_gpu/x_train_' + str(j) + '.npy', x_train)
        np.save('/data/techresearch/Murtaza/mfcc3_gpu/y_train_' + str(j) + '.npy', y_train)
        
    return



create_training_dataset_new()