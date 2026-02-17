const { createCanvas } = require('canvas');
const tf = require('@tensorflow/tfjs-node');

/**
 * Convert canvas to Tensor
 */
function canvasToTensor(canvas) {

    const buffer = canvas.toBuffer('image/png');

    return tf.node.decodeImage(buffer, 1)  // 1 channel (grayscale)
        .resizeNearestNeighbor([40, 100])
        .toFloat()
        .div(255.0); // normalize
}

const CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/**
 * Generate one random character image
 * @returns {{image: Tensor, label: number}}
 */
function generateSample() {

    const canvas = createCanvas(100, 40);
    const ctx = canvas.getContext('2d');

    // White background
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 100, 40);

    // Random character
    const charIndex = Math.floor(Math.random() * CHARS.length);
    const char = CHARS[charIndex];

    ctx.fillStyle = "black";
    ctx.font = "30px Arial";
    ctx.fillText(char, 30, 30);

    return { canvas, label: charIndex };
}


function generateSample2() {

    const canvas = createCanvas(100, 40);
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 100, 40);

    const idx1 = Math.floor(Math.random() * CHARS.length);
    const idx2 = Math.floor(Math.random() * CHARS.length);

    ctx.fillStyle = "black";
    ctx.font = "30px Arial";

    ctx.fillText(CHARS[idx1], 10, 30);
    ctx.fillText(CHARS[idx2], 50, 30);

    return {
        canvas,
        label: [idx1, idx2]
    };
}

function generateBatch2(batchSize) {

    const images = [];
    const labels = [];

    for (let i = 0; i < batchSize; i++) {

        const { canvas, label } = generateSample2();

        const tensor = canvasToTensor(canvas);

        images.push(tensor);

        // Create 72-length label
        const oneHot1 = tf.oneHot([label[0]], 36).arraySync()[0];
        const oneHot2 = tf.oneHot([label[1]], 36).arraySync()[0];

        labels.push([...oneHot1, ...oneHot2]);
    }

    const xs = tf.stack(images);
    const ys = tf.tensor2d(labels);

    return { xs, ys };
}


module.exports = { generateSample, generateSample2, CHARS, canvasToTensor, generateBatch2 };