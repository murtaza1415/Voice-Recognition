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
import numpy as np
from sklearn import preprocessing
import platform
import python_speech_features as mfcc
try:
    import cPickle
except:
    import pickle

"""
code :  This program implemets feature (MFCC + delta)
         extraction process for an audio. 
Note :  20 dim MFCC(19 mfcc coeff + 1 frame log energy)
         20 dim delta computation on MFCC features. 
output : It returns 40 dimensional feature vectors for an audio.
"""



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
   
    
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfilt=20,appendEnergy = True)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta)) 
    return combined
#    
if __name__ == "__main__":
     print("In main, Call extract_features(audio,signal_rate) as parameters")
        
        
def train():
#     import pickle
    import numpy as np
    from scipy.io.wavfile import read
    from sklearn.mixture import GaussianMixture
    #from feature import extract_features
    import warnings
    warnings.filterwarnings("ignore")
    

    #path to training data
    source   = "/home/techresearch/rnn/gmm/gmm data/gmm train/"   

    #path where training speakers will be saved
    dest = "/home/techresearch/rnn/gmm/gmm data/speaker model/"

    train_file = "/home/techresearch/rnn/gmm/gmm data/development_set_enroll.txt"        


    file_paths = open(train_file,'r')

    count = 1

    # Extracting features for each speaker (5 files per speakers)
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()   
        #print path

        # read the audio
        sr,audio = read(source + path)

        # extract 40 dimensional MFCC & delta MFCC features
        vector   = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        # when features of 5 files of speaker are concatenated, then do model training
        if count == 5:    
            gmm = GaussianMixture(n_components = 50, max_iter = 200, covariance_type='diag',n_init = 3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0]+".gmm"
            if platform.python_version()[0] == '3':
                pickle.dump(gmm,open(dest + picklefile,'wb'), protocol=2)
            else:
                cPickle.dump(gmm,open(dest + picklefile,'w'))
            print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
            features = np.asarray(())
            count = 0
        count = count + 1
        

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

            
def train_model():            
    model_path = '/home/techresearch/rnn/gmm/gmm data/speaker model/'
    empty(model_path)
    train()
    return 'Model Trained '

# train_model()