import os
import shutil
import random
import librosa
import numpy as np
from glob import glob
from scipy import signal
from sklearn.svm import SVC
from IPython import display
from IPython.display import SVG
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import load_model
#from keras.utils import model_to_dot
from keras.utils import to_categorical 
from keras.initializers import glorot_uniform
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Add, Dropout, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Dense, Activation, BatchNormalization, ZeroPadding2D
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










def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X




def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X





def ResNet50(input_shape=(512, 299, 1)):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    #X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    fc1 = Conv2D(filters=2048, kernel_size=(16, 1), strides=(1, 1), name = 'fc1')(X)
    fc1 = BatchNormalization(scale=False, axis=3, name = 'bn_fc1')(fc1)
    fc1 = Activation('relu')(fc1) 
    fc1 = Dropout(0.2)(fc1)
    
    apool1 = AveragePooling2D(pool_size=(1, 10), strides=(1,1))(fc1)
    
    flatten = Flatten()(apool1)
    
    fc2 = Dense(5994, activation='relu', name = 'classification')(flatten)
    
    # Create model
    model = Model(inputs = X_input, outputs = fc2, name='ResNet50')

    return model










#Prepare the model.
model = ResNet50()

#model.load_weights('/data/techresearch/Murtaza/vox2/dev/weights4/weights_4_2.h5', by_name=True)
#model.load_weights('/data/techresearch/Murtaza/vox2/dev/weights4/weights_4_10.h5', by_name=True)
#load_custom_weights(model, '/data/techresearch/Murtaza/vox2/dev/weights4/weights_4_2.h5')

#feature_model1 = load_model('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel1_1_40.h5')
#feature_model2 = load_model('/data/techresearch/Murtaza/vox2/dev/weights/featuremodel2_1_40.h5')

adam = keras.optimizers.Adam(lr=0.0001)
#sgd = keras.optimizers.SGD(lr=0.006, momentum=0.9, nesterov=True)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



#run = 4
train_params = {'dim': (512,299),
                'batch_size': 64,
                'n_classes': 5994,
                'n_channels': 1,
                'shuffle': True} 

train_files = glob("/data/techresearch/Murtaza/vox2/dev/wav/*/*.wav")
random.shuffle(train_files)

training_generator = TrainGenerator(train_files, **train_params)

for epoch in range(1, 15):

    print('Batch Size ' + str(train_params['batch_size']))
    print('Epoch: ' + str(epoch))
    model.fit_generator(generator=training_generator,
                        use_multiprocessing=False, workers=8, max_queue_size=10,epochs=1)
    
    model.save('/data/techresearch/Murtaza/vox2/dev/weights_resnet/model_' + '_' + str(epoch) + '.h5')
    #feature_model1.save('/data/techresearch/Murtaza/vox2/dev/weights_resnet/featuremodel1_' + '_' + str(epoch) + '.h5')
    #feature_model2.save('/data/techresearch/Murtaza/vox2/dev/weights_resnet/featuremodel2_' + '_' + str(epoch) + '.h5')
    model.save_weights('/data/techresearch/Murtaza/vox2/dev/weights_resnet/weights_' + '_' + str(epoch+1) + '.h5')  

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
       
              
