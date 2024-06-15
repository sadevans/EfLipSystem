document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Предотвращаем стандартное действие формы (перезагрузку страницы)
    const formData = new FormData(this);
    // console.log(formData);
    fetch('/upload_video/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.video_url) {
            const puploadedVideo = document.getElementById('uploadedVideo');
            uploadedVideo.src = data.video_url;
            document.getElementById('uploadedVideoContainer').style.display = 'block';
        } else {
            console.error('Error:', data.error);
        }
        if (data.filename) {processVideo(data.filename)}
    })
    .catch(error => console.error('Error uploading video:', error));
});


function processVideo(filename){
    fetch(`/process_video/?filename=${filename}`, {
    method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.out_video_url) {
            const puploadedVideo = document.getElementById('processedVideo');
            processedVideo.src = data.out_video_url;
            document.getElementById('processedVideoContainer').style.display = 'block';
        } else {
            console.error('Error:', data.error);
        }
    })
    .catch(error => console.error('Error processing video:', error))

}