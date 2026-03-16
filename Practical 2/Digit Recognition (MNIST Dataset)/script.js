async function trainCNN(){

const output = document.getElementById("output");
output.innerHTML = "Loading MNIST dataset...";


// Load MNIST dataset
const mnist = new MnistData();
await mnist.load();


// Prepare data
const TRAIN_DATA_SIZE = 5500;
const TEST_DATA_SIZE = 1000;

const [trainXs, trainYs] = tf.tidy(() => {
const d = mnist.nextTrainBatch(TRAIN_DATA_SIZE);

return [
d.xs.reshape([TRAIN_DATA_SIZE,28,28,1]),
d.labels
];
});

const [testXs, testYs] = tf.tidy(() => {
const d = mnist.nextTestBatch(TEST_DATA_SIZE);

return [
d.xs.reshape([TEST_DATA_SIZE,28,28,1]),
d.labels
];
});


// CNN Model
const model = tf.sequential();

model.add(tf.layers.conv2d({
inputShape:[28,28,1],
filters:8,
kernelSize:5,
activation:'relu'
}));

model.add(tf.layers.maxPooling2d({
poolSize:2,
strides:2
}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
units:10,
activation:'softmax'
}));


// Compile model
model.compile({
optimizer:'adam',
loss:'categoricalCrossentropy',
metrics:['accuracy']
});

output.innerHTML = "Training CNN model...";


// Train model
await model.fit(trainXs, trainYs,{
epochs:5,
validationSplit:0.15,
callbacks:{
onEpochEnd:(epoch,logs)=>{
console.log("Epoch:",epoch,"Accuracy:",logs.acc);
}
}
});


// Evaluate model
const result = model.evaluate(testXs,testYs);

const accuracy = result[1].dataSync()[0];

output.innerHTML =
"Training Complete <br><br>" +
"Test Accuracy: " + (accuracy*100).toFixed(2) + "%";

}