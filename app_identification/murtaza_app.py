import os
import identify_speaker, verify_speaker, unknown_speaker
import random
from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)


current_sentence = ''
current_speaker = ''
err_message = ''
#match_state = False



#def match_sentence(sentence , path):
#    return True



#def change_sentence():
#    global current_sentence
#    sentence_pool = ['My name is Abhinandan!',
#                     'My voice is my password.',
#                     'I like Biryani!',
#                     'Eat! Sleep! Repeat!',
#                     'Kiki! Do you love me?',
#                     'Your voice is my password.']
#    current_sentence = random.choice(sentence_pool)
#    return

#My voice is my password verify me.
#You can activate security system now.
#My voice is stronger than your passwords.


@app.route('/')
def index():
    #change_sentence()
    current_speaker = 'None'
    return render_template('index.html', speaker = current_speaker)



@app.route('/reset', methods = ['GET', 'POST'])
def reset():
    return redirect(url_for('index'))



@app.route('/result', methods = ['GET', 'POST'])
def result():
    global current_speaker
    return render_template('index.html', speaker = current_speaker)
    
    

#@app.route('/identify')
#def identify():
#    global match_state, current_speaker
#    
#    match_state = match_sentence(sentence = 'My voice is my password.', path='/rnn/app_Murtaza/identify.wav')
#    current_speaker = identify_speaker(path='/rnn/app_Murtaza/identify.wav')
#    
#    return render_template('index.html', display_sentence = current_sentence , speaker = current_speaker)


#def identify():
#    global err_message, current_speaker
#    message_i, gmm_score, filename = identify_speaker.identify()
#    message_v = verify_speaker.speakerVerification()
#    
#    ubm_score = unknown_speaker.ubm_test()
#    os.remove(str(filename))
#    print("GMM Score ", gmm_score, " UBM Score ", ubm_score)
#    ####### Verification Down Under #########
#    if(gmm_score <= ubm_score):
#        text = "Unknown Speaker !!!"
#        err_message = text
#        current_speaker = text
#    else:
#        current_speaker = message_v + message_i
    

    
@app.route('/save_audio', methods = ['GET', 'POST'])
def save_audio():
    global current_speaker
    blob_data = request.files['data']
    audio_data = blob_data.read()
    print(type(audio_data))
    file = open('/home/techresearch/rnn/hello.wav', 'wb')
    file.write(audio_data)
    file.close()
    print('Audio saved!')
    result = identify_speaker.identify('/home/techresearch/rnn/hello.wav')
    current_speaker = result[0]
    return '1'



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6516, ssl_context='adhoc')