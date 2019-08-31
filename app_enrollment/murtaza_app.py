import os
import random
from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)


err_message = ''
current_audio_data = ''

@app.route('/')
def index():
    return render_template('index.html')

 
@app.route('/Enroll', methods = ['POST'])
def enroll():
    global current_audio_data
    #fname = request.form['first-name-input'] 
    #lname = request.form['last-name-input'] 
    eid   = request.form['ID-input'] 
    #file = open('/home/techresearch/rnn/app_enrollment/data/enrollment/' + fname + '_' + lname + '_' + eid + '.wav', 'wb')
    file = open('/home/techresearch/rnn/app_enrollment/data/enrollment/' + eid + '.wav', 'wb')
    file.write(current_audio_data)
    file.close()
    current_audio_data = ''
    return render_template('index.html', message= 'Audio saved for: ' + eid)

    
    

#@app.route('/identify')
#def identify():
#    global match_state, current_speaker
#    
#    match_state = match_sentence(sentence = 'My voice is my password.', path='/rnn/app_Murtaza/identify.wav')
#    current_speaker = identify_speaker(path='/rnn/app_Murtaza/identify.wav')
#    
#    return render_template('index.html', display_sentence = current_sentence , speaker = current_speaker)

'''
def identify():
    global err_message, current_speaker
    message_i, gmm_score, filename = identify_speaker.identify()
    message_v = verify_speaker.speakerVerification()
    
    ubm_score = unknown_speaker.ubm_test()
    os.remove(str(filename))
    print("GMM Score ", gmm_score, " UBM Score ", ubm_score)
    ####### Verification Down Under #########
    if(gmm_score <= ubm_score):
        text = "Unknown Speaker !!!"
        err_message = text
        current_speaker = text
    else:
        current_speaker = message_v + message_i
'''  

'''
@app.route('/save_audio', methods = ['GET', 'POST'])
def save_audio():              
    blob_data = request.files['data']
    audio_data = blob_data.read()
    print(type(audio_data))
    file = open('/home/techresearch/rnn/app_enrollment/enrollment_audio.wav', 'wb')
    file.write(audio_data)
    file.close()
    print('Audio saved!')
    identify()
    return '1'
'''

@app.route('/save_audio', methods = ['GET', 'POST'])
def save_audio():  #This function will only emporarily store audio
    global current_audio_data
    blob_data = request.files['data']
    current_audio_data = blob_data.read()
    print(type(current_audio_data))
    return '1'



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8686, ssl_context='adhoc')