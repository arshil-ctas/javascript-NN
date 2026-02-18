const { createCanvas, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs-node');

// ---------------------------------------------------------------------------
// Character set  (index 0 = blank token for CTC)
// ---------------------------------------------------------------------------
const BLANK = 0;
const CHARS = ['_BLANK_', ...Array.from('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')];
// CHARS[0] = CTC blank, CHARS[1..36] = alphanumeric

function charToIndex(c) {
    const i = CHARS.indexOf(c.toUpperCase());
    return i === -1 ? -1 : i;
}

function indexToChar(i) {
    return i === BLANK ? '' : (CHARS[i] || '');
}

// ---------------------------------------------------------------------------
// Image → Tensor  (grayscale, 32×128, normalised)
// ---------------------------------------------------------------------------
const IMG_H = 32;
const IMG_W = 128;

async function imageTensor(canvas) {
    // Invert if dark-on-light (most OCR models prefer dark bg / light text,
    // but we keep white-bg convention and just normalise)
    const buf = canvas.toBuffer('image/png');
    const t = tf.node.decodeImage(buf, 1)          // [H, W, 1]
        .resizeBilinear([IMG_H, IMG_W])
        .toFloat()
        .div(255.0);
    return t;
}

// Load from file / URL-style Buffer
async function loadImageAsTensor(input) {
    // input can be: Buffer, file path string, or canvas
    if (typeof input === 'string') {
        const img = await loadImage(input);
        const c = createCanvas(img.width, img.height);
        c.getContext('2d').drawImage(img, 0, 0);
        return imageTensor(c);
    }
    if (Buffer.isBuffer(input)) {
        const img = await loadImage(input);
        const c = createCanvas(img.width, img.height);
        c.getContext('2d').drawImage(img, 0, 0);
        return imageTensor(c);
    }
    // Assume it's already a canvas
    return imageTensor(input);
}

// ---------------------------------------------------------------------------
// CTC greedy decode
// ---------------------------------------------------------------------------
function ctcGreedyDecode(logits2d) {
    // logits2d: [timeSteps, numClasses]
    const data = logits2d.arraySync();
    let prev = -1;
    const chars = [];
    for (const step of data) {
        const idx = step.indexOf(Math.max(...step));
        if (idx !== BLANK && idx !== prev) chars.push(indexToChar(idx));
        prev = idx;
    }
    return chars.join('');
}

// ---------------------------------------------------------------------------
// Sample generators
// ---------------------------------------------------------------------------
const ALPHANUM = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

function randomWord(minLen = 1, maxLen = 6) {
    const len = minLen + Math.floor(Math.random() * (maxLen - minLen + 1));
    return Array.from({ length: len }, () => ALPHANUM[Math.floor(Math.random() * ALPHANUM.length)]).join('');
}

// Render a word to canvas with random augmentation
function renderWord(word, augment = true) {
    const canvas = createCanvas(IMG_W * 2, IMG_H * 2); // render big, resize later
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const fontSize = 28 + (augment ? Math.floor(Math.random() * 10) : 0);
    const fonts = augment
        ? ['Arial', 'Courier New', 'Times New Roman', 'Verdana']
        : ['Arial'];
    const font = fonts[Math.floor(Math.random() * fonts.length)];
    ctx.font = `${augment && Math.random() > 0.5 ? 'bold ' : ''}${fontSize}px ${font}`;

    // Random slight rotation
    if (augment) {
        ctx.save();
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate((Math.random() - 0.5) * 0.15);
        ctx.translate(-canvas.width / 2, -canvas.height / 2);
    }

    // Random noise
    if (augment) {
        for (let i = 0; i < 300; i++) {
            ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.15})`;
            ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, 1, 1);
        }
    }

    ctx.fillStyle = augment ? `rgb(${Math.floor(Math.random() * 40)},${Math.floor(Math.random() * 40)},${Math.floor(Math.random() * 40)})` : '#000000';
    ctx.fillText(word, 10, fontSize + 5);

    if (augment) ctx.restore();

    return canvas;
}

// Generate one sample: { canvas, word, labelIndices }
function generateSample(minLen = 1, maxLen = 6) {
    const word = randomWord(minLen, maxLen);
    const canvas = renderWord(word);
    const labelIndices = Array.from(word).map(c => charToIndex(c));
    return { canvas, word, labelIndices };
}

// ---------------------------------------------------------------------------
// Batch for CTC training
// Returns xs [B, H, W, 1] and sparse label info for tf.ctcLoss
// ---------------------------------------------------------------------------
async function generateBatch(batchSize = 32, minLen = 1, maxLen = 6) {
    const tensors = [];
    const sparseIndices = [];
    const sparseValues = [];
    const sequenceLengths = [];
    let maxLabelLen = 0;

    for (let b = 0; b < batchSize; b++) {
        const { canvas, labelIndices } = generateSample(minLen, maxLen);
        const t = await imageTensor(canvas);
        tensors.push(t);
        labelIndices.forEach((v, t2) => {
            sparseIndices.push([b, t2]);
            sparseValues.push(v);
        });
        if (labelIndices.length > maxLabelLen) maxLabelLen = labelIndices.length;
    }

    const xs = tf.stack(tensors);  // [B, H, W, 1]
    return { xs, sparseIndices, sparseValues, batchSize, maxLabelLen };
}

module.exports = {
    CHARS, BLANK, ALPHANUM, IMG_H, IMG_W,
    charToIndex, indexToChar,
    imageTensor, loadImageAsTensor,
    renderWord, generateSample, generateBatch,
    ctcGreedyDecode
};