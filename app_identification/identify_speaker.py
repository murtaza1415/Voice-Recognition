from sklearn.preprocessing import LabelEncoder
import IPython.display as ipd
import re
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
from keras.layers import Input,Flatten
import numpy as np
import random
import numpy
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, RepeatVector, Input, concatenate,Conv2D
from keras.layers import LSTM, Bidirectional, GRU, Activation, TimeDistributed, Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import adam,SGD, Adam, rmsprop, adadelta, adagrad, adamax
from keras.activations import softmax
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from nltk.corpus import stopwords
from numpy import pad
from keras import optimizers
from keras.utils import to_categorical 
import pandas as pd
from numpy import argmax
import h5py
#import tensorflowjs as tfjs
from sklearn.utils import shuffle
import random
import nltk
from keras.models import load_model
from pydub import AudioSegment
import subprocess
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from matplotlib.pyplot import specgram
import platform
from frontend import normfeat
# if platform.python_version()[0] == '3':
#     import pickle
# elif platform.python_version()[0] == '2':
#     import cPickle as pickle
try:
    import cPickle as pickle
except:
    import pickle


"""
code :  This program implemets feature (MFCC + delta)
         extraction process for an audio. 
Note :  20 dim MFCC(19 mfcc coeff + 1 frame log energy)
         20 dim delta computation on MFCC features. 
output : It returns 40 dimensional feature vectors for an audio.
"""

import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc


audio_path = ''


def calculate_delta(array):

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
   
    
#     mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,26,nfilt=26,appendEnergy = True)
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, 20, nfft = 512, appendEnergy = True, preemph=0.0)
#     mfcc_feat = librosa.feature.mfcc(audio,rate)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
#     delta = librosa.feature.delta(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta)) 
    return combined
   
if __name__ == "__main__":
     print("In main, Call extract_features(audio,signal_rate) as parameters")




# def subProcess2():
#     import subprocess
#     name = 'live.wav'
#     #rand = np.random.randint(900)
#     #name = '{}{:03d}.wav'.format(name,rand)
#     #command = "ffmpeg -i '/home/techresearch/rnn/hello.wav' -acodec pcm_s16le -ac 1 -ar 16000 {}".format(name)
#     command = "ffmpeg -y -i '" + audio_path + "' -acodec pcm_s16le -ac 1 -ar 16000 {}".format(name)
#     return subprocess.call(command, shell=True), name

def subProcess2():
    import subprocess
    name = 'live.wav'
    #rand = np.random.randint(900)
    #name = '{}{:03d}.wav'.format(name,rand)
#     command = "ffmpeg -i '/home/techresearch/rnn/hello.wav' -acodec pcm_s16le -ac 1 -ar 16000 {}".format(name)
    command = "ffmpeg -hide_banner -nostats -loglevel fatal -y -i '" + audio_path + "' -vn -acodec pcm_s16le -ac 1 -ar 16000 {}".format(name)

    return subprocess.call(command, shell=True), name



def test():   
    import os
    import numpy as np
    from scipy.io.wavfile import read
    import warnings
    warnings.filterwarnings("ignore")
    import time
    index = 0.0
    #path to test data
    source   = "/home/techresearch/rnn/gmm/gmm data/gmm test/"   

    modelpath = "/home/techresearch/rnn/gmm/gmm data/speaker model 0/"

    test_file = "/home/techresearch/rnn/gmm/gmm data/development_set_test.txt"        

    file_paths = open(test_file,'r')
    
    gmm_files = [os.path.join(modelpath,fname) for fname in 
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    
    
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
                  in gmm_files]
#     print('Speaker: ',speakers)


    for path in file_paths:  
        path = path.strip() 
        audio, sr = librosa.load(source + path, sr=16000)
        vector   = extract_features(audio,sr) 
#         normfeat.cep_sliding_norm(vector)
        normfeat.cmvn(vector)
        log_likelihood = np.zeros(len(models)) 
  
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            #print (gmm)
            scores = np.array(gmm.score(vector))
            #print (scores)
            log_likelihood[i] = scores.sum()
        
        
        winner = np.argmax(log_likelihood)
        
        for val in range(len(log_likelihood)):
            if(speakers[winner] == speakers[val]):
                index = val
        #print('loglikelihood ', winner)
        
        #for val in range(len(log_likelihood)):
        #    print('Log likelihood for identified speaker ',speakers[val]," is ",log_likelihood[val])
        #print "\tdetected as - ", speakers[winner]
        result = speakers[winner]
        result = result.replace(modelpath,"")
        for j in range(900):
            result = result.replace("{:03d}.wav".format(j),"")
#         print("Speaker Identified: ",result) 
        time.sleep(1.0)
        return result, log_likelihood[index]


            
def move2(path,name):
    for subdirs,dirs,files in os.walk(path):
        for file in files:
            if file.startswith(name):
                newAudio = AudioSegment.from_wav(path + file)
                newAudio.export("/home/techresearch/rnn/gmm/gmm data/gmm test/{}".format(file), format="wav")
                save_path = path + file 
                return save_path
            

def empty(folder):
    import os, shutil
    #folder = '/home/techresearch/rnn/gmm/gmm data/speaker model'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)



def identify(path):
    global audio_path
    audio_path = path
    os.chdir('/home/techresearch/rnn/')
    _,identifier = subProcess2()
    test_path = '/home/techresearch/rnn/'
    demo_path = '/home/techresearch/rnn/gmm/gmm data/gmm test/'
    with open('/home/techresearch/rnn/gmm/gmm data/development_set_test.txt', "a") as myfile:        
        open('/home/techresearch/rnn/gmm/gmm data/development_set_test.txt', 'w').close()
        myfile.writelines(str(identifier) + '\n')     
    filename = move2(test_path, identifier)
    speaker, score = test()
#     empty(demo_path)
#     os.remove(str(filename))
#     speaker = speaker.replace("_"," ")
#     print(speaker)
#     print(score)
#     print(filename)
    return speaker, score,  filename
# identify()