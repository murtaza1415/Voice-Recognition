import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
import warnings
warnings.filterwarnings("ignore")
import time
import librosa

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
            
def extract_features(audio,rate):
    
    
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfilt=20,appendEnergy = True)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta)) 
    return combined

def ubm_test():   
    #path to test data
    source   = "/home/techresearch/rnn/gmm/gmm data/gmm test/"   

    modelpath = "/home/techresearch/rnn/gmm/gmm data/ubm model/"

    test_file = "/home/techresearch/rnn/gmm/gmm data/development_set_test.txt"        

    file_paths = open(test_file,'r')


    gmm_files = [os.path.join(modelpath,fname) for fname in 
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    
#     print(gmm_files)
    #Load the Gaussian gender Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    
    # Read the test directory and get the list of test audio files 
    for path in file_paths:   
        path = path.strip() 
        audio, sr = librosa.load(source + path, sr=16000)
        vector   = extract_features(audio,sr)
        ubm = models[0]
        demo_path = '/home/techresearch/rnn/gmm/gmm data/gmm test/'
        empty(demo_path)
        return ubm.score(vector)
        