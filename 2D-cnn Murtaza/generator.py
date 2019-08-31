import os
import h5py
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import psutil
p = psutil.Process()
p.cpu_affinity([0,1,2,5,9,13,17,18,19,20,23,27,30,31])










class TrainGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, list_IDs, batch_size=128, dim=(512,299), n_channels=1,
                     n_classes=5994, shuffle=True):
            'Initialization'
            self.dim = dim
            self.batch_size = batch_size
            self.list_IDs = list_IDs
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.on_epoch_end()

            
        def __len__(self):
            'Denotes the number of batches per epoch'
            '''
            no_batches = 0
            batch_sum = 0
            curr_size = self.batch_size

            while (batch_sum + curr_size) < len(self.list_IDs):
                batch_sum += curr_size
                no_batches += 1
                if no_batches%10 == 0:
                    curr_size += 1
                    
            print('Loop complete...')
            print(no_batches - 2)
            print(int(np.floor(len(self.list_IDs) / self.batch_size)))
            
            return (no_batches - 2)
            '''
            return int(np.floor(len(self.list_IDs) / self.batch_size))

        
        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
            
            '''
            print('Batch size: ' + str(self.batch_size)) 
            if index%10 == 0:
                self.batch_size += 1 
            '''        
            return X, y

        
        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

                
        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size), dtype=int)
            audio_rate = 16000
            no_seconds = 3
            # Generate data
            for i, audio_file in enumerate(list_IDs_temp):
                # Store sample
                audio_data, _ = librosa.load(sr=None, mono=True, path=audio_file)
                
                if len(audio_data)/audio_rate >= no_seconds:
                    start = random.randint(0, len(audio_data) - (audio_rate*no_seconds) )
                    end = start + (audio_rate*no_seconds)
                    audio_clip = audio_data[start : end]
                    image = audio_to_image(audio_clip, audio_rate)
                    image = np.reshape(image, (512,299,1))
                    X[i,] = image

                # Store class
                y[i] = get_label(audio_file)
                #print(y[i])

            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

        
        
        
        
        
        
        
        
        
class ValGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, list_IDs, batch_size=8000, dim=(512,299), n_channels=1,
                     n_classes=5994, shuffle=True):
            'Initialization'
            self.dim = dim
            self.batch_size = batch_size
            self.list_IDs = list_IDs
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.on_epoch_end()

            
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.list_IDs) / self.batch_size))

        
        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

        
        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

                
        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size), dtype=int)
            audio_rate = 16000
            no_seconds = 3
            # Generate data
            for i, audio_file in enumerate(list_IDs_temp):
                # Store sample
                audio_data, _ = librosa.load(sr=None, mono=True, path=audio_file)
                
                if len(audio_data)/audio_rate >= no_seconds:
                    start = random.randint(0, len(audio_data) - (audio_rate*no_seconds) )
                    end = start + (audio_rate*no_seconds)
                    audio_clip = audio_data[start : end]
                    image = audio_to_image(audio_clip, audio_rate)
                    image = np.reshape(image, (512,299,1))
                    X[i,] = image

                # Store class
                y[i] = get_label_val_vox2(audio_file)

            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
      
    
    
    
    
    
    
    
    
    
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
	#signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)

	# truncate to max bucket sizes
	#rsize = max(k for k in buckets if k <= fft_norm.shape[1])
	#rstart = int((fft_norm.shape[1]-rsize)/2)
	#out = fft_norm[:,rstart:rstart+rsize]

	#return out
	return fft_norm










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










def get_label_val_vox2(path):
    '''
    Extract label of audio from 
    the given path.
    '''
    label = path.split('/')[-1]
    label = label.split('_')[-3]
    label = int(label)
    return label










def create_model():
    '''
    Creates and returns a new model.
    '''
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
    conv5 = Dropout(0.25)(conv5)
    mpool5 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv5)
    
    fc6 = Conv2D(filters=4096, kernel_size=(11, 1), strides=(1, 1))(mpool5)
    fc6 = BatchNormalization(scale=False, axis=3)(fc6)
    fc6 = Activation('relu')(fc6) 
    fc6 = Dropout(0.35)(fc6)
    apool6 = AveragePooling2D(pool_size=(1, 5), strides=(1,1))(fc6)
    
    flatten = Flatten()(apool6)
    
    fc7 = Dense(1024, activation='relu')(flatten)
    fc7 = Dropout(0.35)(fc7)
    
    fc8 = Dense(5994, activation='softmax')(fc7)
    #fc8 = Dense(1211, activation='softmax', name='classifcation')(fc7)
    
    model = Model(model_input, fc8)
    feature_model1 = Model(model_input, flatten)
    feature_model2 = Model(model_input, fc7)
    
    
    return model, feature_model1, feature_model2







'''
def load_custom_weights(model, filepath):
    f = h5py.File(filepath, mode='r')
    layer_names = ['dense_2']
    for name in layer_names:
        #print(len(model.get_layer('dense_1').get_weights()[0]))
        #model.layers[28].set_weights(f['dense_2']['dense_2'])
        #print(list(f))
        bias = np.asarray(list(f[name][name]['bias:0']))
        kernel = np.asarray(list(f[name][name]['kernel:0']))
        new_weights = [kernel, bias]
        model.get_layer(name).set_weights(new_weights)
    f.close()
'''



#Prepare the model.
model, feature_model1, feature_model2 = create_model()

#model.load_weights('/data/techresearch/Murtaza/vox2/dev/weights4/weights_4_2.h5', by_name=True)
model.load_weights('/data/techresearch/Murtaza/vox2/dev/weights4/weights_4_26.h5', by_name=True)
#load_custom_weights(model, '/data/techresearch/Murtaza/vox2/dev/weights4/weights_4_2.h5')

#feature_model1 = load_model('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel1_1_40.h5')
#feature_model2 = load_model('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel2_1_40.h5')

#adam = keras.optimizers.Adam(lr=0.0001)
sgd = keras.optimizers.SGD(lr=0.006, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



run = 4
train_params = {'dim': (512,299),
                'batch_size': 190,
                'n_classes': 5994,
                'n_channels': 1,
                'shuffle': True} 

train_files = glob("/data/techresearch/Murtaza/vox2/dev/wav/*/*.wav")
random.shuffle(train_files)

training_generator = TrainGenerator(train_files, **train_params)

for epoch in range(27, 40):

    print('Batch Size ' + str(train_params['batch_size']))
    print('Epoch: ' + str(epoch))
    model.fit_generator(generator=training_generator,
                        use_multiprocessing=True,
                        workers=8, max_queue_size=8, epochs=1)
    
    model.save('/data/techresearch/Murtaza/vox2/dev/weights4/model_' + str(run) + '_' + str(epoch) + '.h5')
    feature_model1.save('/data/techresearch/Murtaza/vox2/dev/weights4/featuremodel1_' + str(run) + '_' + str(epoch) + '.h5')
    feature_model2.save('/data/techresearch/Murtaza/vox2/dev/weights4/featuremodel2_' + str(run) + '_' + str(epoch) + '.h5')
    model.save_weights('/data/techresearch/Murtaza/vox2/dev/weights4/weights_' + str(run) + '_' + str(epoch) + '.h5')  

    #Evaluate
    '''
    x_val = np.load('/data/techresearch/Murtaza/vox2/val/x_val.npy')
    y_val = np.load('/data/techresearch/Murtaza/vox2/val/y_val.npy')
    #Reshape x_val to have 1 channel.
    x_val = np.reshape(x_val, [-1,512,299,1])
    #Onehot encoding for y_val.
    y_val = to_categorical(y_val)
    eval_result = model.evaluate(x=x_val, y=y_val)
    print(eval_result)
    '''
       
              
