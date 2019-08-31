import webrtcvad
from scipy.io import wavfile
import scipy
import sox
import librosa
#  %%time
def VAD(input_file, output_file):
    
    from pydub import AudioSegment

    # fs,audio = wavfile.read('/home/techresearch/rnn/saad_raza_8559_208.wav')

    # a = AudioSegment.from_wav('/home/techresearch/rnn/hello.wav')
#     print(len(audio))
    # audio_n = audio/float(2**15)   #converts each audio frame value to floating pt 
    audio,fs = librosa.load(input_file, sr=16000)
    index=1
    while((len(audio[:index])/float(fs)) != 0.01):
        index+=1
    # print('j value: ',j)
    # wavfile.write('/home/techresearch/rnn/input1.wav', fs, np.int16(audio_n[:j]))
    # librosa.output.write_wav('/home/techresearch/rnn/input1.wav',audio_n[:j],fs)
#     print( len(audio_2))

    vad = webrtcvad.Vad(3)        #creates vad object, with agressiveness set to 3

    def audioSlice(x, fs, framesz, hop):
        framesamp = int(framesz*fs)
        hopsamp = int(hop*fs)
        X = scipy.array([x[i:i+framesamp] for i in range(0, len(x)-framesamp, hopsamp)])
        return X

    framesz=10./1000 #10 ms 
    hop = 1.0*framesz  #10 ms window
    flag = False
    segment = []
    window_duration = 0.01 
    # samples_per_window = int(window_duration * rate + 0.5)
    voice = 1
    silence = 1
    samples_per_window = index
#     cbn = sox.Combiner()

    try:
        for start in np.arange(0, len(audio), samples_per_window):
            stop = min(start + samples_per_window, len(audio))
    #         print('start: ',start,'stop: ',stop)
            Z = audioSlice(audio[stop:], fs, framesz, hop)
    #         print('length: ',len(Z))
        #     Z = audio_n[start:stop]
        #   Z[100] * 32768
    #         fr = np.int16(Z[10]* 32768).tobytes()
            fr = np.int16(Z[10]* 32768).tobytes()
    #         print('length of fr: ',fr)
            if(vad.is_speech(fr, fs)) == True:
                voice+=1
    #             print('voice')
                if flag == False:
    #                 print('first')
                    X1 = audio[start:stop]
        #             wavfile.write('/home/techresearch/rnn/input2.wav',fs,np.int16(X1))
        #             librosa.output.write_wav('/home/techresearch/rnn/input2.wav',X1,fs)
        #             cbn.build(['/home/techresearch/rnn/input1.wav', '/home/techresearch/rnn/input2.wav'], '/home/techresearch/rnn/output.wav', 'concatenate')
                    flag = True
                else:
#                     print('Writing ...')
                    X2 = audio[start:stop]
        #             wavfile.write('/home/techresearch/rnn/input2.wav',fs,np.int16(audio_n[start:stop]))
        #             librosa.output.write_wav('/home/techresearch/rnn/input2.wav',audio_n[start:stop],fs)
                    X1 = np.concatenate((X1,X2),axis=0)
                    librosa.output.write_wav(output_file,X1,fs)
                    command = "ffmpeg -i '{}' -ar 16000 '{}'".format(output_file, output_file)
                    subprocess.call(command, shell=True)
        #             wavfile.write('/home/techresearch/rnn/out.wav',fs,np.int16(X1))
        #             cbn.build(['/home/techresearch/rnn/input2.wav', '/home/techresearch/rnn/output.wav'], '/home/techresearch/rnn/output.wav', 'concatenate')
        #         segment.append(Z)
            else:
                silence+=1
    #             print('silence ')
    except:
        print('VAD performed ')
    print ('voice: ',voice, 'silence: ', silence )
    # Z = audioSlice(audio_n, fs, framesz, hop)
    # print('Z: ',len(Z))
    # fr = np.int16(Z[100] * 32768).tobytes()
    # print('fr: ',len(fr))
    # vad.is_speech(fr, fs)
    # ipd.Audio(data=Z, rate=rate)

    # vad.is_speech(Z[10], fs)