async function runExperiment() {
    const output = document.getElementById('console-output');
    const status = document.getElementById('model-status');

    // --- 1. ELEMENT-WISE OPERATIONS ---
    const a = tf.tensor1d([1, 2, 3, 4]);
    const b = tf.tensor1d([10, 20, 30, 40]);

    const added = a.add(b);       // Element-wise Addition
    const multiplied = a.mul(b);  // Element-wise Multiplication

    // Displaying results in the UI
    output.innerText = `Vector A: [1, 2, 3, 4]\n` +
                       `Vector B: [10, 20, 30, 40]\n\n` +
                       `A + B = [${added.dataSync()}]\n` +
                       `A * B = [${multiplied.dataSync()}]`;

    // --- 2. SYNTHETIC DATA ---
    // Rule: y = 3x + 5
    const xs = tf.tensor2d([0, 1, 2, 3, 4, 5], [6, 1]);
    const ys = tf.tensor2d([5, 8, 11, 14, 17, 20], [6, 1]);

    // --- 3. MODEL CREATION & TRAINING ---
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.sgd(0.01)
    });

    status.innerText = "Training in progress...";

    // Use tfjs-vis to plot the loss
    const container = { name: 'Loss Plot', tab: 'Training' };
    
    await model.fit(xs, ys, {
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(container, ['loss'], { height: 200, callbacks: ['onEpochEnd'] })
    });

    status.innerText = "Training Complete! Testing prediction for X=10...";

    // --- 4. PREDICTION ---
    const prediction = model.predict(tf.tensor2d([10], [1, 1]));
    const val = prediction.dataSync()[0].toFixed(2);
    
    status.innerHTML = `<strong>Done!</strong> Prediction for X=10 is <strong>${val}</strong> (Target: 35)`;
}

// Start the app
runExperiment();