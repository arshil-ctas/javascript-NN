/**
 * predict.js — Use saved model for inference only (no training)
 *
 * Usage:
 *   node predict.js path/to/image.png
 *   node predict.js                     ← runs 20 synthetic tests
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { loadImageAsTensor, generateSample, ctcGreedyDecode, IMG_H, IMG_W, CHARS } = require('./util');
const { loadModel } = require('./model');

const MODEL_DIR = path.join(__dirname, 'saved_model');

async function predict(model, input) {
    const t = await loadImageAsTensor(input);
    const xs = t.expandDims(0);
    const logits = model.predict(xs);
    const logits2d = logits.squeeze([0]);
    const result = ctcGreedyDecode(logits2d);
    [xs, logits, logits2d, t].forEach(x => x.dispose());
    return result;
}

async function main() {
    const model = await loadModel(MODEL_DIR);
    if (!model) {
        console.error('❌ No saved model found. Run: node model.js  to train first.');
        process.exit(1);
    }

    const imagePath = process.argv[2];
    console.log('imagePath: ', imagePath);

    if (imagePath) {
        const fs = await import('fs');
        const image = fs.readFileSync(imagePath);
        console.log('image: ', image);
        const result = await predict(model, image);
        console.log(`\n🔍 Prediction: "${result}"\n`);
    } else {
        console.log('\n📋 Running 20 synthetic tests...\n');
        let correct = 0;
        for (let i = 0; i < 100; i++) {
            const { canvas, word } = generateSample(1, 1);
            const pred = await predict(model, canvas);
            const ok = pred === word;
            if (ok) correct++;
            console.log(`  ${word.padEnd(8)} → ${pred.padEnd(8)} ${ok ? '✓' : '✗'}`);
        }
        console.log(`\n  Accuracy: ${correct}/20 = ${((correct / 100) * 100).toFixed(1)}%\n`);
    }
}

main();