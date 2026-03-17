// --- 1. GENERATE SYNTHETIC DATA ---
const X = [];
const Y = [];
const scatterData = [];
const numSamples = 100;

// True underlying relationship: y = 5x + 3
const trueWeight = 5;
const trueBias = 3;

for (let i = 0; i < numSamples; i++) {
    let x = Math.random(); // Random x between 0 and 1
    let noise = (Math.random() - 0.5) * 1.5; // Random noise
    let y = (trueWeight * x) + trueBias + noise; 
    
    X.push(x);
    Y.push(y);
    scatterData.push({ x: x, y: y }); // Format for Chart.js scatter plot
}

// --- 2. TRAINING FUNCTION (GRADIENT DESCENT) ---
function trainModel(learningRate, epochs) {
    let weight = 0; // Start with random guess 0
    let bias = 0;   // Start with random guess 0
    let lossHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
        let weightGradient = 0;
        let biasGradient = 0;
        let totalLoss = 0;

        for (let i = 0; i < numSamples; i++) {
            let prediction = weight * X[i] + bias;
            let error = prediction - Y[i];
            
            totalLoss += error * error;
            weightGradient += (2 / numSamples) * X[i] * error;
            biasGradient += (2 / numSamples) * error;
        }

        let mse = totalLoss / numSamples;
        
        // Catch exploding gradients (divergence)
        if (mse > 100 || isNaN(mse)) {
            lossHistory.push(null); 
            break;
        } else {
            lossHistory.push(mse);
        }

        weight -= learningRate * weightGradient;
        bias -= learningRate * biasGradient;
    }
    return lossHistory;
}

// --- 3. RUN EXPERIMENTS ---
const epochs = 100;
const epochLabels = Array.from({length: epochs}, (_, i) => i + 1);

const lrTooHigh = trainModel(1.6, epochs);   // Diverges
const lrOptimal = trainModel(0.6, epochs);   // Fast convergence
const lrSlow = trainModel(0.08, epochs);     // Slow
const lrTooSlow = trainModel(0.01, epochs);  // Barely moves

// --- 4. RENDER CHARTS ---

// Chart 1: Scatter Plot of Synthetic Data
const ctxData = document.getElementById('dataChart').getContext('2d');
new Chart(ctxData, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Synthetic Data Points',
            data: scatterData,
            backgroundColor: '#3b82f6'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { title: { display: true, text: 'X input' } },
            y: { title: { display: true, text: 'Y output' } }
        }
    }
});

// Chart 2: Line Chart of Convergence
const ctxLoss = document.getElementById('lossChart').getContext('2d');
new Chart(ctxLoss, {
    type: 'line',
    data: {
        labels: epochLabels,
        datasets: [
            { label: 'LR = 1.6 (Too High)', data: lrTooHigh, borderColor: '#ef4444', fill: false, tension: 0.1 },
            { label: 'LR = 0.6 (Optimal)', data: lrOptimal, borderColor: '#22c55e', fill: false, tension: 0.1 },
            { label: 'LR = 0.08 (Slow)', data: lrSlow, borderColor: '#3b82f6', fill: false, tension: 0.1 },
            { label: 'LR = 0.01 (Very Slow)', data: lrTooSlow, borderColor: '#a855f7', fill: false, tension: 0.1 }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { title: { display: true, text: 'Epochs' } },
            y: { 
                title: { display: true, text: 'Mean Squared Error (Loss)' },
                min: 0, 
                max: 30 
            }
        }
    }
});