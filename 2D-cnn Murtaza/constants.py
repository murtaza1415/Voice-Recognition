from pyaudio import paInt16

# Signal processing
SAMPLE_RATE = 16000
#PREEMPHASIS_ALPHA = 0
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10