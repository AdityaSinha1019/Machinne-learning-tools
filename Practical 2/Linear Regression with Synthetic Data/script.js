async function trainModel() {

const output = document.getElementById("output");
output.innerHTML = "Training started...";


// Generate Synthetic Data
const xs = tf.tensor2d([1,2,3,4,5,6,7,8,9,10], [10,1]);

const ys = tf.tensor2d([
3.1,5.0,7.2,9.1,11.2,13.1,15.3,17.2,19.1,21.0
], [10,1]);


// Build Model using tf.sequential()
const model = tf.sequential();

model.add(tf.layers.dense({
units:1,
inputShape:[1]
}));


// Compile Model
model.compile({
optimizer:'sgd',
loss:'meanSquaredError'
});


// Train Model
await model.fit(xs, ys, {
epochs:200,
callbacks:{
onEpochEnd:(epoch,logs)=>{
console.log("Epoch:",epoch,"Loss:",logs.loss);
}
}
});


// Prediction
const prediction = model.predict(tf.tensor2d([12],[1,1]));

prediction.print();

output.innerHTML =
"Training Complete <br><br>" +
"Prediction for x = 12 : " +
prediction.dataSync()[0].toFixed(2);

}