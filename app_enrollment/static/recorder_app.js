//Tutorial
//https://blog.addpipe.com/using-recorder-js-to-capture-wav-audio-in-your-html5-web-site/


//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
var enrollButton = document.getElementById("enroll-button");
var audioDiv = document.getElementById("audioDiv");
var recordBlip = document.getElementById("recordBlip");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
	console.log("recordButton clicked");

	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    
    var constraints = { "audio": {
                            "mandatory": {
                                "googEchoCancellation": "false",
                                "googAutoGainControl": "false",
                                "googNoiseSuppression": "false",
                                "googHighpassFilter": "false"
                            },
                            "optional": []
                        },
                        video:false }

 	/*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

	recordButton.disabled = true;
	stopButton.disabled = false;
	pauseButton.disabled = false

	/*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device

		*/
		audioContext = new AudioContext();

		//update the format 
		//document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:2, bufferLen:8192})

		//start the recording process
		rec.record()
        recordBlip.style.backgroundColor = "red"
		console.log("Recording started");

	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;
    	pauseButton.disabled = true
	});
}


function pauseRecording(){
	console.log("pauseButton clicked rec.recording=",rec.recording );
	if (rec.recording){
		//pause
		rec.stop();
        recordBlip.style.backgroundColor = "#f9eded"
		pauseButton.innerHTML="Resume";
	}else{
		//resume
		rec.record()
        recordBlip.style.backgroundColor = "red"
		pauseButton.innerHTML="Pause";

	}
}


function stopRecording() {
	console.log("stopButton clicked");

	//disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	pauseButton.disabled = true;

	//reset button just in case the recording is stopped while paused
	pauseButton.innerHTML="Pause";
	
	//tell the recorder to stop the recording
	rec.stop();
    recordBlip.style.backgroundColor = "#f9eded"
	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(displaySaveAudio);
    
}



function displaySaveAudio(blob) {
	
	var url = URL.createObjectURL(blob);
    console.log(url)
	var au = document.createElement('audio');
	// var li = document.createElement('li');
	//var link = document.createElement('a');

	//name of .wav file to use during upload and download (without extendion)
	//var filename = new Date().toISOString();

	//add controls to the <audio> element
	//au.controls = true;
	//au.src = url;
    audioDiv.src = url;
	//add the new audio element to li
	//audioDiv.appendChild(au);

	//add the li element to the ol
	//recordingsList.appendChild(li);
    
    var fd = new FormData();
    console.log(blob)
    fd.append("data",blob);
    $.ajax({
            type: "POST",
            url : "/save_audio",
            data:fd,
            processData: false,
            contentType: false,
            success: ajaxCallback
            });
}



function ajaxCallback(data) {
    enrollButton.disabled = false;
    console.log('Blob transferred! Ajax returned succesfully.')
}

/*
function uploadAudio(blob) {
    var fd = new FormData();
    console.log(blob)
    fd.append("data",blob);
    $.ajax({
            type: "POST",
            url : "/save_audio",
            data:fd,
            processData: false,
            contentType: false,
            success: ajaxCallback
            });
}
*/