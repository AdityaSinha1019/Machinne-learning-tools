let model;

async function loadModel(){

model = await mobilenet.load();

console.log("MobileNet Model Loaded");

}

loadModel();


// Show image preview
document.getElementById("imageUpload").addEventListener("change", function(event){

const reader = new FileReader();

reader.onload = function(){

document.getElementById("preview").src = reader.result;

}

reader.readAsDataURL(event.target.files[0]);

});


async function classifyImage(){

const img = document.getElementById("preview");

const predictions = await model.classify(img);

const result = document.getElementById("result");

result.innerHTML = 
"Prediction: " + predictions[0].className +
"<br>Confidence: " + (predictions[0].probability * 100).toFixed(2) + "%";

}