from __future__ import print_function # In python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from flask import render_template, request, redirect, url_for,flash
import pickle
import numpy as np
from scipy.io.wavfile import read
from app import app
import sys
import wave
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import json
import math
import contextlib
import warnings
from flask_restful import reqparse
warnings.filterwarnings("ignore")
import os
from flask import Flask, jsonify, request
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import logging
#import urllib2
from pydub import AudioSegment
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")
import scipy.io.wavfile as wavfile
import datetime
import subprocess
import uuid
import importlib
from app import identify_speaker
from app import enroll_user
from app import train_speaker
from flask import Flask
# from flask_cors import CORS
unique=""
# app = Flask(__name__)
# application = Flask(__name__)
# CORS(app)



@app.route('/Enroll', methods = ['POST'])
def enroll():
    fname = request.form['first-name-input'] 
    lname = request.form['last-name-input'] 
    eid   = request.form['ID-input'] 
    message = enroll_user.enroll_user_in_db(fname, lname, eid)
    return render_template('allyn_call.html', message=message)



@app.route('/Test', methods = ['POST'])
def test():
    message = identify_speaker.identify()
    return render_template('allyn_call.html', message2=message)


@app.route('/Train', methods = ['POST'])
def train():
    message = train_speaker.train_model()
    return render_template('allyn_call.html', message3=message)


@app.route('/')
# # @cross_origin()
@app.route('/index')
def index():
    return render_template('allyn_call.html')     #allyn.html


@app.route('/SaveAudio', methods = ['POST'])
# @cross_origin()
def SaveAudio():
     try:  
        print ("0")
        fd = request.files['data']
        print ("1")
        f = open('./hello.webm', 'wb')
        print ("2")
        f.write(fd.read())
        print ("3")
        f.close()
        print ('Audio Recieved :', fd)
        unique_filename = str(uuid.uuid4())+".wav"
        unique = unique_filename
#         -hide_banner -loglevel panic
        command = "ffmpeg -nostats -loglevel 0 -i ./hello.webm -acodec pcm_s16le -ac 1 -ar 16000"
        subprocess.call(command, shell=True)
        return "0"
     except:
        print ('Audio Recieved Error:')
        return "1"


if __name__ == "__main__":
    app.run(debug=True)

     