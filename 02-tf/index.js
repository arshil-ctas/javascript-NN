const tf = require('@tensorflow/tfjs-node');
const { generateSample, canvasToTensor, CHARS, generateBatch2, generateSample2 } = require('./util');

/**
 * Create CNN model
 */
function createModel() {

    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [40, 100, 1],
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());

    // Reduced from 64 → 32 (more stable for synthetic dataset)
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 36,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

/**
 * Generate training batch
 */
function generateBatch(batchSize) {

    const images = [];
    const labels = [];

    for (let i = 0; i < batchSize; i++) {

        const { canvas, label } = generateSample();

        const tensor = canvasToTensor(canvas);

        images.push(tensor);
        labels.push(label);
    }

    const xs = tf.stack(images);

    const ys = tf.oneHot(
        tf.tensor1d(labels, 'int32'),
        36
    );

    return { xs, ys };
}

/**
 * Train model with multiple batches per epoch
 */
async function train(model, epochs = 50, batchSize = 128, stepsPerEpoch = 10) {

    for (let epoch = 0; epoch < epochs; epoch++) {

        let totalLoss = 0;
        let totalAcc = 0;

        for (let step = 0; step < stepsPerEpoch; step++) {

            const { xs, ys } = generateBatch(batchSize);

            const history = await model.fit(xs, ys, {
                epochs: 1,
                verbose: 0
            });

            totalLoss += history.history.loss[0];
            totalAcc += history.history.accuracy
                ? history.history.accuracy[0]
                : history.history.acc[0];

            xs.dispose();
            ys.dispose();
        }

        const avgLoss = totalLoss / stepsPerEpoch;
        const avgAcc = totalAcc / stepsPerEpoch;

        console.log(
            `Epoch ${epoch + 1}/${epochs} | Loss: ${avgLoss.toFixed(4)} | Accuracy: ${(avgAcc * 100).toFixed(2)}%`
        );
    }
}

/**
 * Test prediction
 */
async function test(model) {

    const { canvas, label } = generateSample();

    const input = canvasToTensor(canvas).expandDims(0);

    const prediction = model.predict(input);

    const predictedIndex = prediction.argMax(-1).dataSync()[0];

    console.log("\n--- Test Result ---");
    console.log("Actual:    ", CHARS[label]);
    console.log("Predicted: ", CHARS[predictedIndex]);

    input.dispose();
    prediction.dispose();
}

/**
 * Main
 */
async function main() {

    const model = createModel();

    model.summary();

    console.log("\nTraining started...\n");

    await train(model, 50, 128, 10);

    console.log("\nTesting...\n");

    await test(model);

    console.log("\nTraining complete.");
}
// -------------------- Second Model ---------------------

function createModel2() {

    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [40, 100, 1],
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }));

    // 2 characters × 36 classes
    model.add(tf.layers.dense({
        units: 72
    }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

/**
 * Decode prediction to 2-character string
 */
function decodePrediction(predictionTensor) {

    const data = predictionTensor.dataSync();

    const first = data.slice(0, 36);
    const second = data.slice(36, 72);

    const idx1 = first.indexOf(Math.max(...first));
    const idx2 = second.indexOf(Math.max(...second));

    return CHARS[idx1] + CHARS[idx2];
}


async function train2(model, epochs = 30, batchSize = 128, stepsPerEpoch = 10) {

    for (let epoch = 0; epoch < epochs; epoch++) {

        let totalLoss = 0;

        for (let step = 0; step < stepsPerEpoch; step++) {

            const { xs, ys } = generateBatch2(batchSize);

            const history = await model.fit(xs, ys, {
                epochs: 1,
                verbose: 0
            });

            totalLoss += history.history.loss[0];

            xs.dispose();
            ys.dispose();

            // Force GC cleanup of intermediate tensors
            await tf.nextFrame();


        }

        console.log(
            `Epoch ${epoch + 1}/${epochs} | Loss: ${(totalLoss / stepsPerEpoch).toFixed(4)}`
        );
    }
}

/**
 * Test prediction
 */
async function test2(model) {

    const { canvas, label } = generateSample2();

    const input = canvasToTensor(canvas).expandDims(0);

    const prediction = model.predict(input);

    const result = decodePrediction(prediction);

    console.log("\n--- Test Result ---");
    console.log("Actual:    ", CHARS[label[0]] + CHARS[label[1]]);
    console.log("Predicted: ", result);

    input.dispose();
    prediction.dispose();
}

/**
 * Main
 */
async function main2() {

    const model = createModel2();

    model.summary();

    console.log("\nTraining started...\n");

    await train2(model, 80, 128, 10);

    console.log("\nTesting...\n");

    await test2(model);

    console.log("\nDone.");
}

main2();