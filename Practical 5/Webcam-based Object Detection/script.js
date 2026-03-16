let model;
let webcam;

async function loadModel(){

model = await mobilenet.load();
console.log("MobileNet Model Loaded");

}

loadModel();

async function startWebcam(){

const video = document.getElementById("webcam");

const stream = await navigator.mediaDevices.getUserMedia({
video:true
});

video.srcObject = stream;

video.addEventListener("loadeddata", predictFrame);

}


async function predictFrame(){

const video = document.getElementById("webcam");

const prediction = await model.classify(video);

document.getElementById("result").innerHTML =
"Object: " + prediction[0].className +
"<br>Confidence: " + (prediction[0].probability*100).toFixed(2) + "%";

requestAnimationFrame(predictFrame);

}