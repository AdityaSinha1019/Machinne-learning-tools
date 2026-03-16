let model;

async function loadModel(){

model = await toxicity.load(0.9);
console.log("Model Loaded");

}

loadModel();


async function predictSentiment(){

const text = document.getElementById("textInput").value;

const predictions = await model.classify([text]);

let isNegative = false;

predictions.forEach(p => {

if(p.results[0].match === true){
isNegative = true;
}

});

const result = document.getElementById("result");

if(isNegative){

result.innerHTML = "Sentiment: Negative 😞";

}else{

result.innerHTML = "Sentiment: Positive 😊";

}

}