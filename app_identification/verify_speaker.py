def speakerVerification():
    from nltk.translate.bleu_score import sentence_bleu
    import os
    import subprocess
    
    os.chdir('/home/techresearch/rnn/deepspeech-venv/')
    command = "deepspeech --model /home/techresearch/rnn/deepspeech-venv/PhoneticallyRichSentences/output_graph_phonetically.pbmm --alphabet /home/techresearch/rnn/deepspeech-venv/models/alphabet.txt --lm /home/techresearch/rnn/deepspeech-venv/models/lm.binary --trie /home/techresearch/rnn/deepspeech-venv/models/trie --audio /home/techresearch/rnn/hello.wav > /home/techresearch/rnn/deepspeech-venv/speech.txt"
    subprocess.call(command, shell = True)
    file =  open('/home/techresearch/rnn/deepspeech-venv/speech.txt','r') 
    data =  file.readlines()
    predicted = data.pop()
    predicted = str(predicted).split()
    actual = str("you can get in without your password").split()
    print(predicted)
    print(actual)
    actual = [actual]
    score  = sentence_bleu(actual, predicted) * 100
    if ( score >= 50.0 ):
        return 'Speaker Verified as '
        print('BLEU-1 Score: ', score)
    elif(score < 50.0):
        return 'Speaker Not Verified, Try Again! '
        print('BLEU-1 Score: ', score)