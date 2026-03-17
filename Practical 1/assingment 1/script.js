async function runRegression() {
    const status = document.getElementById('status-bar');

    // --- 1. GENERATE SYNTHETIC DATA ---
    // Function: y = 2x + 5 (with some randomness)
    const generateData = (numPoints) => {
        return tf.tidy(() => {
            const xs = tf.randomUniform([numPoints], 0, 10);
            const ys = xs.mul(2).add(5).add(tf.randomNormal([numPoints], 0, 1));
            return {
                inputs: xs,
                labels: ys
            };
        });
    };

    const data = generateData(100);
    const inputs = data.inputs.arraySync();
    const labels = data.labels.arraySync();

    // Format data for plotting
    const values = inputs.map((x, i) => ({ x, y: labels[i] }));

    // Plot initial data
    tfvis.render.scatterplot(
        { name: 'Original Data vs Predictions' },
        { values: [values], series: ['Original'] },
        { xLabel: 'Input (X)', yLabel: 'Label (Y)', height: 400 }
    );

    // --- 2. BUILD THE MODEL ---
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: 'meanSquaredError'
    });

    // --- 3. TRAIN THE MODEL ---
    status.innerText = "Status: Training... watch the chart update!";
    
    await model.fit(data.inputs.reshape([100, 1]), data.labels.reshape([100, 1]), {
        epochs: 50,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Periodically plot predictions to show the line "moving"
                if (epoch % 10 === 0) {
                    const testXs = tf.linspace(0, 10, 10);
                    const preds = model.predict(testXs.reshape([10, 1]));
                    const predArray = await preds.array();
                    const linePoints = testXs.arraySync().map((x, i) => ({ x, y: predArray[i][0] }));
                    
                    tfvis.render.scatterplot(
                        { name: 'Original Data vs Predictions' },
                        { values: [values, linePoints], series: ['Original', 'Model Prediction'] },
                        { xLabel: 'Input (X)', yLabel: 'Label (Y)', height: 400 }
                    );
                }
            }
        }
    });

    status.innerText = "Status: Training Complete! The line now fits the data.";
    status.style.background = "#e6ffed";
    status.style.color = "#22863a";
}

runRegression();