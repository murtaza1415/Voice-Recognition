import os
import shutil
import random
import librosa
import numpy as np
from glob import glob
from scipy import signal
from sklearn.svm import SVC
from IPython import display
from pydub import AudioSegment
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization
from python_speech_features import mfcc

#For fft spectrum.
import sigproc
import constants as c
from scipy.signal import lfilter, butter


#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import psutil
p = psutil.Process()
p.cpu_affinity([0,1,2,5,9,13,17,18,19,20,23,27,30,31])

from pydub import AudioSegment



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




def vox2_iterative_generative_training(model, feature_model1, feature_model2):
    '''
    
    '''
    
    audio_files = glob("/data/techresearch/Murtaza/vox2/dev/wav/*/*.wav")
    #random.shuffle(audio_files)
    run = 3
    no_epochs = 15
    subset_size = 10000
    total_subsets = len(audio_files)//subset_size
    audio_rate = 16000
    no_seconds = 3
    batch_size = 128
    
    for epoch in range(0,no_epochs):
        
        random.shuffle(audio_files)
        
        for j in range(0,total_subsets):
            x_train = []
            y_train = []
            
            audio_subset = audio_files[ (subset_size*j) : (subset_size*(j+1)) ]
            print('Subset: ' + str((subset_size*j)) + ' to ' + str(subset_size*(j+1)) + '   Batch_size: ' + str(batch_size))

            for audio_file in audio_subset:
                label = get_label(audio_file)
                #audio_data, _ = librosa.load(sr=audio_rate, mono=True, path=audio_file)
                audio_data, _ = librosa.load(sr=None, mono=True, path=audio_file)
                #audio_data = AudioSegment.from_file(audio_file)
                #audio_data = audio_data.get_array_of_samples()
                #audio_data = np.array(audio_data)

                #assert _ == audio_rate

                if len(audio_data)/audio_rate >= no_seconds:
                    start = random.randint(0, len(audio_data) - (audio_rate*no_seconds) )
                    end = start + (audio_rate*no_seconds)
                    audio_clip = audio_data[start : end]
                    image = audio_to_image(audio_clip, audio_rate)
                    x_train.append(image)
                    y_train.append(label)

            #Convert to numpy
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            
            #Reshape x_train to have 1 channel.
            x_train = np.reshape(x_train, [-1,512,299,1])
            #Onehot encoding for y_train.
            y_train = to_categorical(y_train, num_classes=5994)
            
            print('Epoch: ' + str(epoch))
            model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1, validation_split=0.03)
            batch_size += 5
            
            if (j+1)%40 == 0:
                
                x_train = np.load('/data/techresearch/Murtaza/vox2/val/x_val.npy')
                y_train = np.load('/data/techresearch/Murtaza/vox2/val/y_val.npy')
                #Reshape x_train to have 1 channel.
                x_train = np.reshape(x_train, [-1,512,299,1])
                #Onehot encoding for y_train.
                y_train = to_categorical(y_train)
                eval_result = model.evaluate(x=x_train, y=y_train) 
                
                file = open('train_log.txt', 'a')
                file.write('Run: ' + str(run) + '\n')
                file.write('Epoch: ' + str(epoch + 1) + '\n')
                file.write(str((subset_size*j)) + ' to ' + str(subset_size*(j+1)) + '\n')
                file.write('Batch size: ' + str(batch_size) + '\n')
                file.write(str(eval_result))
                file.write('\n\n')
                file.close()
                
                model.save('/data/techresearch/Murtaza/vox2/dev/weights/model_' + str(run) + '_' + str(epoch+1) + '_' + str(j+1) + '.h5')
                feature_model1.save('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel1_' + str(run) + '_' + str(epoch+1) + '_' + str(j+1) + '.h5')
                feature_model2.save('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel2_' + str(run) + '_' + str(epoch+1) + '_' + str(j+1) + '.h5')
                model.save_weights('/data/techresearch/Murtaza/vox2/dev/weights/weights_' + str(run) + '_' + str(epoch+1) + '_' + str(j+1) + '.h5')             
    
    return







def create_model():
    '''
    Creates and returns a new model.
    '''
    
    #For Spectogram Input features (No convergence)
    '''
    model_input = Input(shape=(129,285,1))
    
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu')(model_input)
    mpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv1)
    
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), activation='relu')(mpool1)
    #mpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv2)
    
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv4)
    
    flatten = Flatten()(conv5)
    
    fc7 = Dense(1024)(flatten)
    fc8 = Dense(1200, activation='softmax')(fc7)
    
    model = Model(model_input, fc8)
    '''
    
    
    #For MFCC1 input features (Acc: 84, val_acc:70)
    '''
    model_input = Input(shape=(60,126,1))
    
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu')(model_input)
    mpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv1)
    
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), activation='relu')(mpool1)
    #mpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv2)
    
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv2)
    #conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv3)
    #conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv4)
    
    flatten = Flatten()(conv3)
    
    fc7 = Dense(1024)(flatten)
    fc8 = Dense(1200, activation='softmax')(fc7)
    
    model = Model(model_input, fc8)
    '''
    
    #For mfcc2 Input features (Good Convegence was witnessed)
    '''
    model_input = Input(shape=(332,300,1))
    
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu')(model_input)
    mpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv1)
    
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), activation='relu')(mpool1)
    mpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv2)
    
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(mpool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv4)
    
    flatten = Flatten()(conv5)
    
    fc7 = Dense(1024)(flatten)
    fc8 = Dense(1200, activation='softmax')(fc7)
    
    model = Model(model_input, fc8)
    '''
    
    
    #For mfcc3 Input features

    model_input = Input(shape=(512,299,1))
    #model_input_1 = BatchNormalization(scale=False, axis=3)(model_input)
    
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2))(model_input)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1) 
    mpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv1)
    
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2))(mpool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2) 
    mpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv2)
    
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1))(mpool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(conv3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4) 
    
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(conv4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.15)(conv5)
    mpool5 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv5)
    
    fc6 = Conv2D(filters=4096, kernel_size=(11, 1), strides=(1, 1))(mpool5)
    fc6 = BatchNormalization(scale=False, axis=3)(fc6)
    fc6 = Activation('relu')(fc6) 
    fc6 = Dropout(0.2)(fc6)
    apool6 = AveragePooling2D(pool_size=(1, 5), strides=(1,1))(fc6)
    
    flatten = Flatten()(apool6)
    
    fc7 = Dense(1024, activation='relu')(flatten)
    fc7 = Dropout(0.2)(fc7)
    fc8 = Dense(5994, activation='softmax')(fc7)
    
    model = Model(model_input, fc8)
    feature_model1 = Model(model_input, flatten)
    feature_model2 = Model(model_input, fc7)
    
    
    return model, feature_model1, feature_model2





#dataset_to_wav1()


#Prepare the model.
model, feature_model1, feature_model2 = create_model()
modl.load_weights('/data/techresearch/Murtaza/vox2/dev/weights/weights_1_120.h5', by_name=True)
#feature_model1 = load_model('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel1_1_40.h5')
#feature_model2 = load_model('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel2_1_40.h5')

#adam = keras.optimizers.Adam(lr=0.0001)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



vox2_iterative_generative_training(model, feature_model1, feature_model2)