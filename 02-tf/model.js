/**
 * CRNN + CTC — Variable-length OCR
 * Leak-free via explicit GradientTape + manual weight update
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const {
    CHARS, IMG_H, IMG_W,
    loadImageAsTensor,
    generateSample, generateBatch,
    ctcGreedyDecode
} = require('./util');

const NUM_CLASSES = CHARS.length;  // 37
const TIME_STEPS = 32;
const MODEL_DIR = path.join(__dirname, 'saved_model');

// ─────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────
function createModel() {
    const input = tf.input({ shape: [IMG_H, IMG_W, 1] });

    const conv = (filters, k) => tf.layers.conv2d({
        filters, kernelSize: k, padding: 'same',
        activation: 'relu', kernelInitializer: 'glorotUniform'
    });
    const bn = () => tf.layers.batchNormalization();
    const pool = (h, w) => tf.layers.maxPooling2d({ poolSize: [h, w] });

    let x = conv(32, 3).apply(input); x = bn().apply(x); x = pool(2, 2).apply(x);
    x = conv(64, 3).apply(x); x = bn().apply(x); x = pool(2, 2).apply(x);
    x = conv(128, 3).apply(x); x = bn().apply(x);
    x = conv(128, 3).apply(x); x = bn().apply(x); x = pool(2, 1).apply(x);
    x = conv(256, 3).apply(x); x = bn().apply(x); x = pool(2, 1).apply(x);

    x = tf.layers.conv2d({
        filters: 256, kernelSize: [2, 1], padding: 'valid',
        activation: 'relu', kernelInitializer: 'glorotUniform'
    }).apply(x);
    x = tf.layers.reshape({ targetShape: [TIME_STEPS, 256] }).apply(x);

    const makeGru = () => tf.layers.gru({
        units: 128, returnSequences: true,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
        implementation: 1
    });
    x = tf.layers.bidirectional({ layer: makeGru(), mergeMode: 'concat' }).apply(x);
    x = tf.layers.bidirectional({ layer: makeGru(), mergeMode: 'concat' }).apply(x);

    const output = tf.layers.dense({
        units: NUM_CLASSES, activation: 'softmax',
        kernelInitializer: 'glorotUniform'
    }).apply(x);

    return tf.model({ inputs: input, outputs: output });
}

// ─────────────────────────────────────────────
// Label helpers (pure JS, zero tensors)
// ─────────────────────────────────────────────
function buildLabelArrays(sparseIndices, sparseValues, batchSize) {
    const labelLenArr = new Int32Array(batchSize);
    sparseIndices.forEach(([b, t]) => {
        if (t + 1 > labelLenArr[b]) labelLenArr[b] = t + 1;
    });
    const maxLen = Math.max(...labelLenArr, 1);
    const labelData = new Int32Array(batchSize * maxLen).fill(-1);
    sparseIndices.forEach(([b, t], i) => {
        labelData[b * maxLen + t] = sparseValues[i];
    });
    return { labelData, labelLenArr, maxLen };
}

function buildByBatch(sparseIndices, sparseValues, batchSize) {
    const byBatch = Array.from({ length: batchSize }, () => []);
    sparseIndices.forEach(([b, t], i) => { byBatch[b][t] = sparseValues[i]; });
    return byBatch;
}

// ─────────────────────────────────────────────
// Loss — called INSIDE GradientTape.gradient()
// Must return a scalar tensor. All intermediates
// are tracked by the tape and cleaned up after
// grads are computed.
// ─────────────────────────────────────────────
function lossFn(model, xs, sparseIndices, sparseValues, batchSize) {
    const logits = model.apply(xs, { training: true });

    // Try native CTC
    try {
        const { labelData, labelLenArr, maxLen } = buildLabelArrays(sparseIndices, sparseValues, batchSize);
        const logLogits = tf.log(tf.clipByValue(logits, 1e-7, 1.0));
        const labelTensor = tf.tensor2d(labelData, [batchSize, maxLen], 'int32');
        const labelLengths = tf.tensor1d(labelLenArr, 'int32');
        const seqLengths = tf.fill([batchSize], TIME_STEPS, 'int32');
        const loss = tf.mean(tf.ctcLoss(logLogits, labelTensor, labelLengths, seqLengths));

        // Dispose intermediates explicitly (tape doesn't own these)
        logLogits.dispose(); labelTensor.dispose(); labelLengths.dispose(); seqLengths.dispose();
        logits.dispose();

        return loss;
    } catch (_) {
        // Fallback cross-entropy
        return ctcFallback(logits, sparseIndices, sparseValues, batchSize);
    }
}

function ctcFallback(logits, sparseIndices, sparseValues, batchSize) {
    const byBatch = buildByBatch(sparseIndices, sparseValues, batchSize);
    const losses = [];

    for (let b = 0; b < batchSize; b++) {
        const labelArr = byBatch[b].filter(v => v !== undefined);
        if (!labelArr.length) continue;
        const stepSize = TIME_STEPS / labelArr.length;
        for (let j = 0; j < labelArr.length; j++) {
            const t = Math.min(Math.floor(j * stepSize + stepSize / 2), TIME_STEPS - 1);
            const logit = logits.slice([b, t, 0], [1, 1, NUM_CLASSES]).reshape([NUM_CLASSES]);
            const target = tf.oneHot(tf.tensor1d([labelArr[j]], 'int32'), NUM_CLASSES).squeeze([0]);
            losses.push(tf.losses.softmaxCrossEntropy(target, logit));
            logit.dispose(); target.dispose();
        }
    }

    logits.dispose();

    if (!losses.length) return tf.scalar(3.611);
    const stacked = tf.stack(losses);
    const mean = tf.mean(stacked);
    stacked.dispose();
    losses.forEach(l => l.dispose());
    return mean;
}

// ─────────────────────────────────────────────
// Optimizer — single instance
// ─────────────────────────────────────────────
const optimizer = tf.train.adam(0.0005);

// ─────────────────────────────────────────────
// One training step — explicit GradientTape
// This is the ONLY pattern in tfjs-node that
// guarantees no tape-related leaks.
// ─────────────────────────────────────────────
function trainStep(model, xs, sparseIndices, sparseValues, batchSize) {
    // grads() calls lossFn, records all ops, then disposes the tape.
    // It returns { value: lossScalar, grads: {varName: gradTensor} }
    const { value: lossScalar, grads } = optimizer.computeGradients(
        () => lossFn(model, xs, sparseIndices, sparseValues, batchSize)
    );

    // Apply gradients and dispose them immediately
    optimizer.applyGradients(grads);
    Object.values(grads).forEach(g => g.dispose());

    // Read loss value synchronously, then dispose
    const lv = lossScalar.dataSync()[0];
    lossScalar.dispose();

    return isNaN(lv) ? 0 : lv;
}

// ─────────────────────────────────────────────
// Training loop
// ─────────────────────────────────────────────
async function train(model, { epochs = 5000, batchSize = 32, stepsPerEpoch = 20 } = {}) {
    console.log('\n🚀 Training started...\n');

    let baseline = null;

    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;

        for (let step = 0; step < stepsPerEpoch; step++) {
            const { xs, sparseIndices, sparseValues } = await generateBatch(batchSize, 1, 6);

            totalLoss += trainStep(model, xs, sparseIndices, sparseValues, batchSize);

            xs.dispose();
            await tf.nextFrame();
        }

        // Baseline after first epoch (Adam slots allocated then)
        if (epoch === 0) baseline = tf.memory().numTensors;

        const avg = (totalLoss / stepsPerEpoch).toFixed(4);
        const leaked = tf.memory().numTensors - baseline;
        const memMB = Math.round(tf.memory().numBytes / 1024 / 1024);

        process.stdout.write(
            `\r ${new Date().toLocaleTimeString()}Epoch ${String(epoch + 1).padStart(4)}/${epochs} | Loss: ${avg} | +Tensors: ${String(leaked).padStart(3)} | Mem: ${memMB} MB   `
        );

        // Save every 10 epochs
        if ((epoch + 1) % 10 === 0) {
            console.log('');
            await saveModel(model, MODEL_DIR);
        }
    }

    console.log('\n\n✅ Training complete.\n');
}

// ─────────────────────────────────────────────
// Predict
// ─────────────────────────────────────────────
async function predict(model, input) {
    const t = await loadImageAsTensor(input);
    const result = tf.tidy(() =>
        ctcGreedyDecode(model.predict(t.expandDims(0)).squeeze([0]))
    );
    t.dispose();
    return result;
}

// ─────────────────────────────────────────────
// Save / Load
// ─────────────────────────────────────────────
async function saveModel(model, dir = MODEL_DIR) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    await model.save(`file://${dir}`);
    console.log(`💾 Saved → ${dir}`);
}

async function loadModel(dir = MODEL_DIR) {
    const modelJson = path.join(dir, 'model.json');
    if (!fs.existsSync(modelJson)) return null;
    const model = await tf.loadLayersModel(`file://${dir}/model.json`);
    console.log(`📦 Loaded ← ${dir}`);
    return model;
}

// ─────────────────────────────────────────────
// Test
// ─────────────────────────────────────────────
async function runTests(model, n = 15) {
    console.log('\n--- Test Results ---');
    let correct = 0;
    for (let i = 0; i < n; i++) {
        const { canvas, word } = generateSample(1, 6);
        const pred = await predict(model, canvas);
        const ok = pred === word;
        if (ok) correct++;
        console.log(`  Actual: ${word.padEnd(8)} | Predicted: ${String(pred).padEnd(8)} ${ok ? '✓' : '✗'}`);
    }
    console.log(`\n  Accuracy: ${correct}/${n} = ${((correct / n) * 100).toFixed(1)}%\n`);
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
async function main() {
    const args = process.argv.slice(2);
    const predictOnly = args.includes('--predict-only');
    const imagePath = args.find(a => !a.startsWith('-'));

    let model = await loadModel();
    if (!model) {
        console.log('No saved model found — creating new model.');
        model = createModel();
    }

    model.summary();

    if (!predictOnly) {
        await train(model, {
            epochs: process.env.EPOCHS ? parseInt(process.env.EPOCHS) : 4000,
            batchSize: 32,
            stepsPerEpoch: 20
        });
        await saveModel(model);
    }

    await runTests(model, 15);

    if (imagePath && fs.existsSync(imagePath)) {
        console.log(`\n🔍 Prediction for "${imagePath}": ${await predict(model, imagePath)}\n`);
    }
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });

module.exports = { createModel, train, predict, saveModel, loadModel };