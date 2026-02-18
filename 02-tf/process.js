/**
 * preprocess.js
 * 
 * Utilities to clean up real-world images before feeding to the model:
 * - Signature canvas screenshots (dark strokes on white)
 * - Photos of handwritten text
 * - Scanned documents
 * - Screenshots of typed text
 *
 * Usage:
 *   const { preprocessImage } = require('./preprocess');
 *   const cleanBuffer = await preprocessImage(inputBuffer);
 *   // then pass cleanBuffer to predict()
 */

const { createCanvas, loadImage } = require('canvas');

/**
 * Main preprocessing pipeline:
 * 1. Convert to grayscale
 * 2. Otsu-style binarisation (auto threshold)
 * 3. Crop to content bounding box + padding
 * 4. Return as PNG Buffer (ready for model)
 */
async function preprocessImage(input) {
    let img;
    if (Buffer.isBuffer(input) || typeof input === 'string') {
        img = await loadImage(input);
    } else {
        img = input; // already an HTMLImageElement / canvas
    }

    // Draw to working canvas
    const src = createCanvas(img.width, img.height);
    const ctx = src.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, src.width, src.height);
    const { data, width, height } = imageData;

    // --- Grayscale ---
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    // --- Otsu threshold ---
    const threshold = otsuThreshold(gray);

    // --- Binarise (text = 0 / black, background = 255 / white) ---
    for (let i = 0; i < gray.length; i++) {
        gray[i] = gray[i] < threshold ? 0 : 255;
    }

    // --- Find bounding box of dark pixels ---
    let minX = width, maxX = 0, minY = height, maxY = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (gray[y * width + x] === 0) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    // If nothing found, return original
    if (minX > maxX || minY > maxY) {
        return src.toBuffer('image/png');
    }

    const pad = 8;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(width - 1, maxX + pad);
    maxY = Math.min(height - 1, maxY + pad);

    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;

    // --- Write clean image ---
    const out = createCanvas(cropW, cropH);
    const octx = out.getContext('2d');
    octx.fillStyle = '#ffffff';
    octx.fillRect(0, 0, cropW, cropH);

    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            const v = gray[(y + minY) * width + (x + minX)];
            octx.fillStyle = `rgb(${v},${v},${v})`;
            octx.fillRect(x, y, 1, 1);
        }
    }

    return out.toBuffer('image/png');
}

/** Otsu's thresholding */
function otsuThreshold(pixels) {
    const hist = new Array(256).fill(0);
    pixels.forEach(p => hist[p]++);
    const total = pixels.length;

    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];

    let sumB = 0, wB = 0, max = 0, threshold = 128;

    for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB === 0) continue;
        const wF = total - wB;
        if (wF === 0) break;
        sumB += t * hist[t];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const between = wB * wF * (mB - mF) ** 2;
        if (between > max) {
            max = between;
            threshold = t;
        }
    }
    return threshold;
}

module.exports = { preprocessImage, otsuThreshold };