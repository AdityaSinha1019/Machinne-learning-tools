async function runDemo() {
    const output = document.getElementById('tensor-output');
    const status = document.getElementById('model-status');

    // --- 1. RESHAPE & FLATTEN DEMONSTRATION ---
    // Start with a 1D Vector of 8 elements
    const original = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);
    
    // Reshape into a 2x4 Matrix
    const reshaped = original.reshape([2, 4]);
    
    // Flatten it back to 1D
    const flattened = reshaped.flatten();

    output.innerText = 
        `Original (1D): [${original.shape}]\n` +
        `Reshaped (2x4):\n${reshaped.toString()}\n\n` +
        `Flattened (1D): [${flattened.shape}]`;

    // --- 2. SYNTHETIC DATA (Non-Linear Pattern) ---
    // Let's train the model to learn y = x^2 (Parabola)
    const xValues = [-3, -2, -1, 0, 1, 2, 3];
    const yValues = xValues.map(x => x * x);

    const xs = tf.tensor2d(xValues, [7, 1]);
    const ys = tf.tensor2d(yValues, [7, 1]);

    // --- 3. MODEL ARCHITECTURE ---
    const model = tf.sequential();
    
    // For non-linear data, we need at least one hidden layer with activation
    model.add(tf.layers.dense({units: 8, inputShape: [1], activation: 'relu'}));
    model.add(tf.layers.dense({units: 1}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.1) // Adam is better for non-linear patterns
    });

    status.innerText = "Training model to learn square numbers...";

    // Show training visuals
    await model.fit(xs, ys, {
        epochs: 150,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Charts' },
            ['loss']
        )
    });

    // --- 4. PREDICTION ---
    const testVal = 4;
    const prediction = model.predict(tf.tensor2d([testVal], [1, 1]));
    const result = prediction.dataSync()[0].toFixed(2);

    status.innerHTML = `Training Complete! <br> Predicted square of ${testVal}: <strong>${result}</strong> (Target: 16)`;
    status.style.color = "#27ae60";
}

runDemo();