let recorder;
let recordedChunks = []

const videoLiveElement = document.getElementById('liveVideo');
const uploadButton = document.getElementById('uploadButton');
const videoFile = document.getElementById('videoFile');
const uploadForm = document.getElementById('uploadForm');

const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');
let frameRate; 
let isStreaming = false;


const videoFrames = document.getElementById('videoFrames');
// const eventSource = new EventSource('/give_frames/');


uploadButton.addEventListener('click', () => {
    videoFile.value = null;
    videoFile.click();
});

videoFile.addEventListener('change', () => {
    if (videoFile.files.length > 0) {

        // const videoLiveElement = document.getElementById('liveVideo');
        const stream = videoLiveElement.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            document.getElementById('liveVideoContainer').style.display = 'none';
            document.getElementById('liveVideo').style.display = 'none';

            videoLiveElement.srcObject = null;
        }

        const formData = new FormData(uploadForm);
        formData.append('video', videoFile.files[0]);

        fetch('/upload_video/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to upload video: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.video_url) {
                const uploadedVideo = document.getElementById('uploadedVideo');
                document.getElementById('processedVideoContainer').style.display = 'none';
                uploadedVideo.src = data.video_url;
                document.getElementById('uploadedVideo').style.display = 'block';
                document.getElementById('uploadedVideoContainer').style.display = 'block';
            } else {
                console.error('Error:', data.error);
            }

            if (data.filename) {
                processVideo(data.filename);
            }
        })
        .catch(error => {
            console.error('Error uploading video:', error);
        });
    } else {
        console.log("No file selected for upload.");
    }
});

function processVideo(filename){
    console.log('IN PROCESS VIDEO')
    fetch(`/process_video/?filename=${filename}`, {
    method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.out_video_url) {
            const processedVideo = document.getElementById('processedVideo');
            processedVideo.src = data.out_video_url;
            document.getElementById('processedVideo').style.display = 'block';
            document.getElementById('processedVideoContainer').style.display = 'block';
        } else {
            console.error('Error:', data.error);
        }
    })
    .catch(error => console.error('Error processing video:', error))

}

async function startVideoPreview() {
    document.getElementById('uploadedVideo').style.display = 'none';
    document.getElementById('processedVideo').style.display = 'none';
    document.getElementById('uploadedVideoContainer').style.display = 'none';
    document.getElementById('processedVideoContainer').style.display = 'none';
    try {
        // if (stream) {
        //     stream.getTracks().forEach(track => track.stop());
        // }
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true});
        const track = stream.getVideoTracks()[0];
        const settings = track.getSettings();
        frameRate = settings.frameRate;
        console.log(frameRate)
        const videoElement = document.getElementById('liveVideo');
        videoElement.srcObject = stream;
        videoElement.play();

        recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (event) => {
            recordedChunks.push(event.data);
        };

        recorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            sendBlobToServer(blob);
        };
        document.getElementById('liveVideo').style.display = 'block';
        document.getElementById('liveVideoContainer').style.display = 'block';
        document.getElementById('startRecord').style.display = 'inline-block';
        document.getElementById('endRecord').style.display = 'inline-block';

    } catch (error) {
        console.error('Ошибка при доступе к камере:', error);
    }
}

const recordButton = document.getElementById('recordButton');
recordButton.addEventListener('click', startVideoPreview);

document.getElementById('startRecord').addEventListener('click', () => {
    if (recorder && recorder.state === 'inactive') {
        recordedChunks = [];
        recorder.start();
    }
});

document.getElementById('endRecord').addEventListener('click', () => {
    if (recorder && recorder.state === 'recording') {
        recorder.stop();
    }
});

function sendBlobToServer(blob) {
    var csrftoken = getCookie('csrftoken');
    const formData = new FormData();
    formData.append('video', blob, 'recorded-video.webm');
    formData.append('frameRate', frameRate);

    fetch('/get_video/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.filename) {
            processVideo(data.filename);
        } else {
            console.error('Error:', data.message);
        }
    })
    .catch(error => {
        console.error('Upload failed:', error);
    });
}


function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


function startStreaming() {
    console.log(isStreaming)
    console.log(frameRate)
    if (isStreaming) return; 
    isStreaming = true;
    console.log('IN START STREAMING ');
  
    setInterval(() => {
      if (isStreaming) {
        canvas.width = videoLiveElement.videoWidth;
        canvas.height = videoLiveElement.videoHeight;
        context.drawImage(videoLiveElement, 0, 0, canvas.width, canvas.height);
        const frameDataURL = canvas.toDataURL('image/jpeg'); 
        sendFrameToServer(frameDataURL); 
      }
    }, 1000 / frameRate);
  }
  
function stopStreaming() {
    var csrftoken = getCookie('csrftoken');
    isStreaming = false;
    console.log('IN STOP STREAMING');
    fetch('/change_stream_flag/', { // Adjust URL
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
        },
        body: 0
    })
    .then(response => response.json())
    .then(data => {
        console.log('Server response:', data); 
    })
    .catch(error => {
        console.error('Error sending frame:', error); 
    });
}



function sendFrameToServer(frameDataURL) {
    var csrftoken = getCookie('csrftoken');
    const data = {
        image: frameDataURL,
        fps: frameRate
      };
    fetch('/receive_frame/', { // Adjust URL
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Server response:', data); 
    })
    .catch(error => {
        console.error('Error sending frame:', error); 
    });
}

// document.getElementById('startStream').addEventListener('click', startStreaming);
// document.getElementById('stopStream').addEventListener('click', stopStreaming);