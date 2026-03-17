// --- 1. TENSOR DIMENSIONS (Scalar, Vector, Matrix) ---
console.log("%c--- Tensor Dimensions ---", "color: blue; font-weight: bold;");

// Scalar (0D Tensor)
const scalar = tf.scalar(3.14);
scalar.print(); 

// Vector (1D Tensor)
const vector = tf.tensor1d([1, 2, 3]);
vector.print();

// Matrix (2D Tensor)
const matrix = tf.tensor2d([[1, 2], [3, 4]]);
matrix.print();

console.log("Check the shapes above: Scalar (empty), Vector (3), Matrix (2,2)");


// --- 2. SYNTHETIC DATA GENERATION ---
// We want the model to learn: y = 2x - 1
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);


// --- 3. CREATE AND TRAIN THE MODEL ---
async function trainModel() {
    const status = document.getElementById('status');
    status.innerText = "Status: Training model... please wait.";

    // Define a simple linear model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: specify loss and optimizer
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd' // Stochastic Gradient Descent
    });

    // Visualizing the training using tfjs-vis
    const surface = { name: 'Model Loss', tab: 'Training' };
    
    // Train the model
    await model.fit(xs, ys, {
        epochs: 100,
        callbacks: tfvis.show.fitCallbacks(surface, ['loss'])
    });

    status.innerText = "Status: Training Complete!";
    status.style.color = "green";

    // --- 4. PREDICTION ---
    // Try to predict y if x is 10 (Expected: 2*10 - 1 = 19)
    const result = model.predict(tf.tensor2d([10], [1, 1]));
    console.log("%cPrediction for X=10:", "color: green; font-weight: bold;");
    result.print();
}

// Run the training
trainModel();