async function runValidation() {
    const status = document.getElementById('status');
    const table = document.getElementById('results-table');
    const tableBody = document.getElementById('table-body');

    // --- 1. TRAINING DATA (The "Known" world) ---
    // Rule: y = 10x - 5
    const trainX = tf.tensor2d([0, 1, 2, 3, 4, 5], [6, 1]);
    const trainY = tf.tensor2d([-5, 5, 15, 25, 35, 45], [6, 1]);

    // --- 2. DEFINE MODEL ---
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: 'meanSquaredError'
    });

    // Train the model
    await model.fit(trainX, trainY, {
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks({ name: 'Training', tab: 'Log' }, ['loss'])
    });

    status.innerText = "Status: Training Complete. Predicting unseen values...";
    status.classList.add('success');
    table.style.display = 'table';

    // --- 3. UNSEEN TEST DATA ---
    // These values were NOT in the training set
    const unseenInputs = [10, 25, 50, -2]; 

    unseenInputs.forEach(x => {
        // Expected result based on y = 10x - 5
        const expected = (10 * x) - 5;

        // Model prediction
        const predictionTensor = model.predict(tf.tensor2d([x], [1, 1]));
        const prediction = predictionTensor.dataSync()[0];

        // Calculate how close it got (Percentage)
        const accuracy = (1 - Math.abs((expected - prediction) / expected)) * 100;

        // Append to table
        const row = `<tr>
            <td>${x}</td>
            <td>${expected}</td>
            <td>${prediction.toFixed(2)}</td>
            <td>${accuracy.toFixed(2)}%</td>
        </tr>`;
        tableBody.innerHTML += row;
    });
}

runValidation();