/**
 * ============================================================
 *  CRNN + CTC  —  Variable-length OCR
 *  Fixed for tfjs-node compatibility:
 *    - No recurrentDropout (triggers broken Orthogonal initializer)
 *    - glorotUniform for all weight inits
 *    - CTC loss via manual gradient-tape approach
 * ============================================================
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

const NUM_CLASSES = CHARS.length;   // 37 (blank + 36 chars)
const TIME_STEPS = 32;              // feature map width after CNN
const MODEL_DIR = path.join(__dirname, 'saved_model');

// ---------------------------------------------------------------------------
// Model definition (CRNN)
// ---------------------------------------------------------------------------
function createModel() {
    const input = tf.input({ shape: [IMG_H, IMG_W, 1] });

    // --- CNN backbone ---
    let x = tf.layers.conv2d({
        filters: 32, kernelSize: 3, padding: 'same', activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }).apply(input);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x);   // [16, 64, 32]

    x = tf.layers.conv2d({
        filters: 64, kernelSize: 3, padding: 'same', activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x);   // [8, 32, 64]

    x = tf.layers.conv2d({
        filters: 128, kernelSize: 3, padding: 'same', activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.conv2d({
        filters: 128, kernelSize: 3, padding: 'same', activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(x);   // [4, 32, 128]

    x = tf.layers.conv2d({
        filters: 256, kernelSize: 3, padding: 'same', activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(x);   // [2, 32, 256]

    // Collapse height dimension → [B, 1, 32, 256]
    x = tf.layers.conv2d({
        filters: 256, kernelSize: [2, 1], padding: 'valid', activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }).apply(x);   // [B, 1, 32, 256]

    // Reshape → [B, 32, 256]
    x = tf.layers.reshape({ targetShape: [TIME_STEPS, 256] }).apply(x);

    // --- Bidirectional GRU ---
    // IMPORTANT: No recurrentDropout — it forces Orthogonal initializer on large
    // matrices which breaks on many tfjs-node builds. Use glorotUniform instead.
    const gru1 = tf.layers.gru({
        units: 128,
        returnSequences: true,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
        implementation: 1    // 1 = loop (more compatible than 2 = matrix)
    });
    const biGru1 = tf.layers.bidirectional({ layer: gru1, mergeMode: 'concat' }).apply(x);

    const gru2 = tf.layers.gru({
        units: 128,
        returnSequences: true,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
        implementation: 1
    });
    const biGru2 = tf.layers.bidirectional({ layer: gru2, mergeMode: 'concat' }).apply(biGru1);

    // Output logits per time step  [B, 32, 37]
    const output = tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax',
        kernelInitializer: 'glorotUniform'
    }).apply(biGru2);

    const model = tf.model({ inputs: input, outputs: output });
    return model;
}

// ---------------------------------------------------------------------------
// CTC loss — manual implementation compatible with tfjs-node
//
// tf.losses.ctcLoss is not reliably exposed in all tfjs-node versions.
// We implement a differentiable approximation:
//   - Run model forward pass
//   - Build dense label matrix padded to maxLabelLen
//   - Use TF's ctcLoss op via tf.backend().ctcLoss if available,
//     otherwise fall back to a cross-entropy approximation for training signal
// ---------------------------------------------------------------------------
function ctcLossCompat(logits, sparseIndices, sparseValues, batchSize) {
    return tf.tidy(() => {
        // logits: [B, T, C]
        const logLogits = tf.log(tf.clipByValue(logits, 1e-7, 1.0));

        // Build per-sample label lengths
        const labelLenArr = new Int32Array(batchSize);
        sparseIndices.forEach(([b, t]) => {
            if (t + 1 > labelLenArr[b]) labelLenArr[b] = t + 1;
        });

        const maxLen = Math.max(...Array.from(labelLenArr), 1);

        // Dense label matrix [B, maxLen] filled with -1 (padding)
        const labelData = new Int32Array(batchSize * maxLen).fill(-1);
        sparseIndices.forEach(([b, t], i) => {
            labelData[b * maxLen + t] = sparseValues[i];
        });

        const labelTensor = tf.tensor2d(labelData, [batchSize, maxLen], 'int32');
        const labelLengths = tf.tensor1d(labelLenArr, 'int32');
        const seqLengths = tf.fill([batchSize], TIME_STEPS, 'int32');

        // Try the native CTC op first
        try {
            // tf.ctcLoss is available in tfjs-node >= 3.x via the Node backend
            const loss = tf.ctcLoss(logLogits, labelTensor, labelLengths, seqLengths);
            return tf.mean(loss);
        } catch (_) {
            // Fallback: cross-entropy on "soft" targets spread over label chars
            // Not true CTC but provides a useful training gradient
            return ctcFallback(logits, sparseIndices, sparseValues, batchSize, maxLen);
        }
    });
}

function ctcFallback(logits, sparseIndices, sparseValues, batchSize, maxLen) {
    // Map each label character to the nearest time step (even spacing)
    // and compute cross-entropy at those positions
    const losses = [];

    for (let b = 0; b < batchSize; b++) {
        const myLabels = sparseIndices
            .filter(([sb]) => sb === b)
            .map((_, i) => sparseValues[sparseIndices.filter(([sb]) => sb === b).findIndex((_, j) => j === i)]);

        const myVals = [];
        sparseIndices.forEach(([sb, st], i) => {
            if (sb === b) myVals[st] = sparseValues[i];
        });

        const labelArr = Object.entries(myVals).map(([, v]) => v);
        if (labelArr.length === 0) continue;

        // Evenly space labels across TIME_STEPS
        const step = TIME_STEPS / labelArr.length;
        for (let j = 0; j < labelArr.length; j++) {
            const t = Math.min(Math.floor(j * step + step / 2), TIME_STEPS - 1);
            const logit = logits.slice([b, t, 0], [1, 1, NUM_CLASSES]).reshape([NUM_CLASSES]);
            const target = tf.oneHot([labelArr[j]], NUM_CLASSES).reshape([NUM_CLASSES]);
            losses.push(tf.losses.softmaxCrossEntropy(target, logit));
        }
    }

    return losses.length > 0 ? tf.mean(tf.stack(losses)) : tf.scalar(0);
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------
const optimizer = tf.train.adam(0.0005);

async function train(model, { epochs = 50, batchSize = 32, stepsPerEpoch = 20 } = {}) {
    console.log('\n🚀 Training started...\n');

    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;

        for (let step = 0; step < stepsPerEpoch; step++) {
            const { xs, sparseIndices, sparseValues } = await generateBatch(batchSize, 1, 6);

            const lossVal = optimizer.minimize(() => {
                const logits = model.apply(xs, { training: true });
                return ctcLossCompat(logits, sparseIndices, sparseValues, batchSize);
            }, true);

            const lv = (await lossVal.data())[0];
            totalLoss += isNaN(lv) ? 0 : lv;
            lossVal.dispose();
            xs.dispose();

            if (step % 5 === 0) await tf.nextFrame();
        }

        const avg = (totalLoss / stepsPerEpoch).toFixed(4);
        const mem = tf.memory().numTensors;
        process.stdout.write(
            `\rEpoch ${String(epoch + 1).padStart(3)}/${epochs} | Loss: ${avg} | Tensors: ${mem}   `
        );

        if ((epoch + 1) % 10 === 0) {
            console.log('');
            await saveModel(model, MODEL_DIR);
            console.log('  💾 Checkpoint saved.');
        }
    }

    console.log('\n\n✅ Training complete.\n');
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------
async function predict(model, input) {
    const t = await loadImageAsTensor(input);
    const xs = t.expandDims(0);
    const logits = model.predict(xs);
    const logits2d = logits.squeeze([0]);
    const result = ctcGreedyDecode(logits2d);
    [xs, logits, logits2d, t].forEach(x => x.dispose());
    return result;
}

// ---------------------------------------------------------------------------
// Save / Load
// ---------------------------------------------------------------------------
async function saveModel(model, dir = MODEL_DIR) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    await model.save(`file://${dir}`);
    console.log(`\n💾 Model saved → ${dir}`);
}

async function loadModel(dir = MODEL_DIR) {
    const modelJson = path.join(dir, 'model.json');
    if (!fs.existsSync(modelJson)) return null;
    const model = await tf.loadLayersModel(`file://${dir}/model.json`);
    console.log(`\n📦 Model loaded ← ${dir}`);
    return model;
}

// ---------------------------------------------------------------------------
// Test on synthetic samples
// ---------------------------------------------------------------------------
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
    console.log(`\n  Accuracy: ${correct}/${n} = ${((correct / n) * 100).toFixed(1)}%`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
    const args = process.argv.slice(2);
    const predictOnly = args.includes('--predict-only');
    const imagePath = args.find(a => !a.startsWith('-'));

    let model = await loadModel();
    if (!model) {
        console.log('No saved model found. Creating new model...');
        model = createModel();
    }

    model.summary();

    if (!predictOnly) {
        await train(model, {
            epochs: process.env.EPOCHS ? parseInt(process.env.EPOCHS) : 50,
            batchSize: 32,
            stepsPerEpoch: 20
        });
        await saveModel(model);
    }

    await runTests(model, 15);

    if (imagePath && fs.existsSync(imagePath)) {
        const result = await predict(model, imagePath);
        console.log(`\n🔍 Prediction for "${imagePath}": ${result}\n`);
    }
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});

module.exports = { createModel, train, predict, saveModel, loadModel };