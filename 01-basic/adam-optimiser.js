// =============================================================================
//  FROM PERCEPTRON TO LANGUAGE MODEL — Pure JavaScript, No Libraries
//  A learning journey: MLP → Embeddings → Softmax → Next-Token Prediction → Adam
// =============================================================================
//
//  WHAT THIS FILE TEACHES (in order):
//
//  PART 0 — Math Primitives (matrix ops you'll need everywhere)
//  PART 1 — Your Original MLP, generalized & cleaned
//  PART 2 — Softmax + Cross-Entropy (how to predict from a vocabulary)
//  PART 3 — Embeddings (turning tokens into learnable vectors)
//  PART 4 — Bigram Language Model (simplest possible next-token predictor)
//  PART 5 — Adam Optimizer (WHY it exists, HOW it works, line by line)
//  PART 6 — MLP Language Model with Adam wired in
//  PART 7 — Conceptual Map: What's left to become a real Transformer
//
// =============================================================================




// =============================================================================
//  PART 0 — MATH PRIMITIVES
//
//  Every neural network, no matter how complex, is just:
//    - Matrix × vector multiplications
//    - Nonlinear activations (tanh, relu, softmax)
//    - Gradient descent (adjust params to reduce loss)
//
//  These helpers are used in EVERY part of this file.
// =============================================================================

const M = {

    // Create a 2D matrix filled with zeros: shape [rows x cols]
    zeros: (rows, cols) =>
        Array.from({ length: rows }, () => new Float32Array(cols)),

    // Random matrix with Xavier initialization
    //   WHY Xavier? Random weights that are too big → activations explode
    //               Random weights that are too small → gradients vanish
    //   Xavier picks a scale based on layer size so neither happens.
    //   Formula: scale = sqrt(2 / (fan_in + fan_out))
    rand: (rows, cols, scale) => {
        scale = scale ?? Math.sqrt(2.0 / (rows + cols));
        return Array.from({ length: rows }, () =>
            Float32Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
        );
    },

    // Matrix × vector: [rows x cols] · [cols] → [rows]
    //   This is literally the core operation of every dense/linear layer.
    //   output[i] = sum over j of (W[i][j] * x[j])
    matVec: (W, x) => {
        const out = new Float32Array(W.length);
        for (let i = 0; i < W.length; i++) {
            let s = 0;
            for (let j = 0; j < x.length; j++) s += W[i][j] * x[j];
            out[i] = s;
        }
        return out;
    },

    // Vector dot product: sum of element-wise products
    dot: (a, b) => {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    },

    // Add a bias vector to an array in-place and return it
    addBias: (z, b) => {
        for (let i = 0; i < z.length; i++) z[i] += b[i];
        return z;
    },

    // Tanh activation — smooth, outputs between -1 and +1
    //   Better than sigmoid for hidden layers: gradients don't die as fast
    tanh: (z) => z.map(Math.tanh),

    // Softmax: turns raw scores (logits) into a probability distribution
    //
    //   WHY subtract max before exp?
    //   If logits = [1000, 1001], then exp(1000) = Infinity in JS.
    //   Subtracting max makes it exp(0) and exp(1) — perfectly safe.
    //   The result is mathematically identical, just numerically stable.
    //
    softmax: (logits) => {
        const maxL = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxL));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    },

    // Cross-entropy loss: -log(probability assigned to the correct answer)
    //
    //   WHY log? Because probability alone doesn't punish hard enough.
    //   If correct prob = 0.99 → loss = 0.01  (tiny penalty, correct)
    //   If correct prob = 0.01 → loss = 4.60  (huge penalty, wrong)
    //   log makes the model pay heavily for being confidently wrong.
    //
    crossEntropy: (probs, targetIndex) => -Math.log(probs[targetIndex] + 1e-9),

    // Combined gradient of softmax + cross-entropy loss
    //
    //   WHY combine? Because the math simplifies beautifully:
    //   dLoss/dLogit[i] = probs[i] - 1   (if i == target)
    //   dLoss/dLogit[i] = probs[i]        (if i != target)
    //
    //   This is the most elegant result in neural net calculus.
    //   You don't need to separately differentiate softmax and cross-entropy.
    //
    softmaxCrossEntropyGrad: (probs, targetIndex) => {
        const grad = Float32Array.from(probs);
        grad[targetIndex] -= 1.0;
        return grad;
    },
};




// =============================================================================
//  PART 1 — THE MLP (Multi-Layer Perceptron)
//
//  Your original network, but now:
//   - Uses softmax output (predicts from a class set, not just 0/1)
//   - Uses tanh activation (more stable than sigmoid for hidden layers)
//   - Uses the clean math primitives above
//
//  Architecture: inputs → [W1, b1, tanh] → [W2, b2, softmax] → probs
// =============================================================================

class MLP {

    /**
     * @param {number} inputSize  - number of input features
     * @param {number} hiddenSize - neurons in the hidden layer
     * @param {number} outputSize - number of output classes (vocab size)
     * @param {number} lr         - learning rate
     */
    constructor(inputSize, hiddenSize, outputSize, lr = 0.05) {
        this.lr = lr;

        // Layer 1 weights + biases: input → hidden
        this.W1 = M.rand(hiddenSize, inputSize);
        this.b1 = new Float32Array(hiddenSize);     // start at zero

        // Layer 2 weights + biases: hidden → output (logits)
        this.W2 = M.rand(outputSize, hiddenSize);
        this.b2 = new Float32Array(outputSize);
    }

    // Run the network forward and return probabilities
    forward(x) {
        this.x = x;
        this.z1 = M.addBias(M.matVec(this.W1, x), this.b1);  // pre-activation
        this.a1 = M.tanh(this.z1);                             // post-activation
        this.logits = M.addBias(M.matVec(this.W2, this.a1), this.b2);
        this.probs = M.softmax(this.logits);
        return this.probs;
    }

    // Forward pass + backprop + weight update in one call
    train(x, targetIndex) {
        const probs = this.forward(x);
        const loss = M.crossEntropy(probs, targetIndex);

        // --- Backpropagation ---
        // Think of it as: "how much did each weight contribute to the error?"
        // We compute gradients layer by layer, from output back to input.

        // Gradient at the output (softmax + cross-entropy combined)
        const dLogits = M.softmaxCrossEntropyGrad(probs, targetIndex);

        // Layer 2 gradients
        const dW2 = M.zeros(this.W2.length, this.a1.length);
        const db2 = new Float32Array(this.b2.length);
        const da1 = new Float32Array(this.a1.length);

        for (let i = 0; i < this.W2.length; i++) {
            db2[i] = dLogits[i];
            for (let j = 0; j < this.a1.length; j++) {
                dW2[i][j] = dLogits[i] * this.a1[j];
                da1[j] += dLogits[i] * this.W2[i][j];  // chain rule: accumulate
            }
        }

        // Through tanh: derivative is (1 - tanh(z)^2)
        //   WHY? Because d/dz[tanh(z)] = 1 - tanh(z)^2
        //   We already computed tanh(z) = a1, so: d = (1 - a1^2)
        const dz1 = this.z1.map((z, i) => da1[i] * (1 - Math.tanh(z) ** 2));

        // Layer 1 gradients
        const dW1 = M.zeros(this.W1.length, this.x.length);
        const db1 = new Float32Array(this.b1.length);

        for (let i = 0; i < this.W1.length; i++) {
            db1[i] = dz1[i];
            for (let j = 0; j < this.x.length; j++) {
                dW1[i][j] = dz1[i] * this.x[j];
            }
        }

        // --- Gradient Descent Update ---
        // The simplest possible update: move each weight a tiny step
        // against the gradient (downhill on the loss surface)
        for (let i = 0; i < this.W1.length; i++) {
            this.b1[i] -= this.lr * db1[i];
            for (let j = 0; j < this.x.length; j++) {
                this.W1[i][j] -= this.lr * dW1[i][j];
            }
        }
        for (let i = 0; i < this.W2.length; i++) {
            this.b2[i] -= this.lr * db2[i];
            for (let j = 0; j < this.a1.length; j++) {
                this.W2[i][j] -= this.lr * dW2[i][j];
            }
        }

        return loss;
    }
}




// =============================================================================
//  PART 2 — SOFTMAX + CROSS-ENTROPY DEMO
//
//  Your original network predicted 0 or 1 (binary sigmoid).
//  Language models predict from a VOCABULARY (could be 50,000 tokens).
//
//  Softmax = "which class is most likely, expressed as probabilities?"
//  Cross-entropy = "how badly did we do at predicting the right class?"
// =============================================================================

function demonstrateSoftmax() {
    console.log("\n=== PART 2: Softmax + Cross-Entropy Demo ===");

    // Imagine our vocab is 4 characters: a, b, c, d
    // The model outputs these raw scores (logits):
    const logits = [2.0, 0.5, -1.0, 0.1];
    const probs = M.softmax(logits);

    console.log("Logits (raw scores):  ", Array.from(logits));
    console.log("Probs  (after softmax):", Array.from(probs).map(p => p.toFixed(3)));
    console.log("Sum of probs:          ", probs.reduce((a, b) => a + b, 0).toFixed(6), "(always 1.0)");
    console.log("");
    console.log("If correct answer is 'a' (index 0): loss =", M.crossEntropy(probs, 0).toFixed(4), "← low, model is right");
    console.log("If correct answer is 'b' (index 1): loss =", M.crossEntropy(probs, 1).toFixed(4), "← high, model is wrong");
    console.log("");
    console.log("The gradient (dLoss/dLogits) when target is 'b':");
    console.log("  ", Array.from(M.softmaxCrossEntropyGrad(probs, 1)).map(g => g.toFixed(3)));
    console.log("  Notice: probs[1] - 1 = negative (push it up), all others = positive (push down)");
}
demonstrateSoftmax();




// =============================================================================
//  PART 3 — EMBEDDINGS
//
//  Problem: tokens are discrete (characters, words). Neural nets need numbers.
//
//  Bad solution: one-hot encoding
//    'a' → [1, 0, 0, 0, 0, ...]   (vocabSize long, mostly zeros)
//    'b' → [0, 1, 0, 0, 0, ...]
//    Problem: no relationship encoded. 'a' and 'b' are equally "different"
//             from each other as 'a' and 'z'. The network has to learn everything.
//
//  Good solution: Embedding table
//    'a' → [0.12, -0.34, 0.87, ...]   (embeddingDim floats, dense & learned)
//    'b' → [0.15, -0.31, 0.83, ...]   (similar chars end up with similar vectors!)
//
//  The embedding table is just a matrix [vocabSize x embeddingDim].
//  Looking up token index 3 = grabbing row 3 of that matrix.
//  Those rows are parameters — they get updated by the optimizer just like weights.
//
//  This is EXACTLY what GPT uses. The only difference is scale.
// =============================================================================

class EmbeddingTable {

    /**
     * @param {number} vocabSize    - total number of unique tokens
     * @param {number} embeddingDim - how many floats represent each token
     */
    constructor(vocabSize, embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;

        // The actual table: [vocabSize x embeddingDim]
        // Small random init — these will be learned during training
        this.table = M.rand(vocabSize, embeddingDim, 0.1);
    }

    // Lookup a single token → returns its embedding vector (a row)
    lookup(tokenIndex) {
        return this.table[tokenIndex];
    }

    // Lookup multiple tokens and concatenate their embeddings
    //   e.g. context [h=3, e=7, l=2] → [emb(3), emb(7), emb(2)] concatenated
    //   Result length = contextSize * embeddingDim
    lookupAndConcat(tokenIndices) {
        const out = new Float32Array(tokenIndices.length * this.embeddingDim);
        for (let i = 0; i < tokenIndices.length; i++) {
            out.set(this.table[tokenIndices[i]], i * this.embeddingDim);
        }
        return out;
    }
}




// =============================================================================
//  PART 4 — BIGRAM LANGUAGE MODEL
//
//  The simplest possible "what comes next?" model.
//  Given one character, predict the next one.
//
//  Architecture:
//    prevChar → EmbeddingTable → Linear(W, b) → Softmax → nextChar probs
//
//  Why is this called "bigram"?
//  A bigram is a pair of adjacent elements. We only look at pairs (prev, next).
//  No history beyond one character.
//
//  This is exactly where Andrej Karpathy's "makemore" series starts.
// =============================================================================

class BigramLM {

    constructor(vocab, embeddingDim = 8, lr = 0.05) {
        this.vocab = vocab;
        this.vocabSize = vocab.length;
        this.charToIdx = Object.fromEntries(vocab.map((c, i) => [c, i]));
        this.lr = lr;

        // Each character gets a learned embedding vector
        this.embed = new EmbeddingTable(this.vocabSize, embeddingDim);

        // Project embedding → logits over vocab
        this.W = M.rand(this.vocabSize, embeddingDim);
        this.b = new Float32Array(this.vocabSize);
    }

    forward(prevCharIdx) {
        this.prevIdx = prevCharIdx;
        this.emb = this.embed.lookup(prevCharIdx);
        this.logits = M.addBias(M.matVec(this.W, this.emb), this.b);
        this.probs = M.softmax(this.logits);
        return this.probs;
    }

    train(prevCharIdx, nextCharIdx) {
        const probs = this.forward(prevCharIdx);
        const loss = M.crossEntropy(probs, nextCharIdx);

        const dLogits = M.softmaxCrossEntropyGrad(probs, nextCharIdx);

        const dEmb = new Float32Array(this.emb.length);

        for (let i = 0; i < this.W.length; i++) {
            this.b[i] -= this.lr * dLogits[i];
            for (let j = 0; j < this.emb.length; j++) {
                this.W[i][j] -= this.lr * dLogits[i] * this.emb[j];
                dEmb[j] += dLogits[i] * this.W[i][j];
            }
        }

        // Gradient flows back into the embedding row for this character
        const row = this.embed.table[this.prevIdx];
        for (let d = 0; d < row.length; d++) {
            row[d] -= this.lr * dEmb[d];
        }

        return loss;
    }

    generate(startChar, length = 30) {
        let idx = this.charToIdx[startChar] ?? 0;
        let out = startChar;
        for (let i = 0; i < length; i++) {
            const probs = this.forward(idx);
            idx = sampleFromProbs(probs);
            out += this.vocab[idx];
        }
        return out;
    }
}




// =============================================================================
//  PART 5 — ADAM OPTIMIZER
//
//  This is the single biggest practical upgrade you can make to a neural network.
//  Every major model today (GPT, BERT, LLaMA) trains with Adam or a variant of it.
//
//  THE PROBLEM WITH PLAIN GRADIENT DESCENT:
//  ─────────────────────────────────────────
//  θ -= lr * grad
//
//  One learning rate for every parameter. But different parameters need different
//  step sizes:
//    - Embedding for rare characters → small gradients, need bigger steps
//    - Output bias → large gradients, needs smaller steps
//    - Some parameters oscillate (gradient keeps flipping sign) → need smaller steps
//    - Some parameters point consistently in one direction → can take bigger steps
//
//  Adam solves this by tracking TWO statistics per parameter, updated every step:
//
//  ─────────────────────────────────────────────────────────────────────────────
//
//  m  = "first moment" = MOMENTUM = smoothed average of recent gradients
//
//       m = β1 * m + (1 - β1) * grad
//
//       β1 is typically 0.9 — so m is 90% previous momentum, 10% new gradient.
//
//       WHY? It smooths out noisy gradients. If the gradient oscillates between
//       +0.5 and -0.5, momentum stays near zero → tiny step. If it's consistently
//       +0.5, momentum builds to +0.5 → bigger step. Like a ball rolling downhill:
//       it builds speed in a consistent direction but doesn't thrash around.
//
//  ─────────────────────────────────────────────────────────────────────────────
//
//  v  = "second moment" = VARIANCE = smoothed average of recent gradient SQUARED
//
//       v = β2 * v + (1 - β2) * grad²
//
//       β2 is typically 0.999 — v changes very slowly.
//
//       WHY squared? Because v measures the "size" of recent gradients.
//       If gradients are consistently large → v is large → divide → smaller step
//       If gradients are small/noisy     → v is small → divide → larger step
//       This is the "adaptive" part of Adam: each parameter gets its own lr.
//
//  ─────────────────────────────────────────────────────────────────────────────
//
//  BIAS CORRECTION — the part everyone forgets to explain:
//
//       m̂ = m / (1 - β1^t)
//       v̂ = v / (1 - β2^t)
//
//       At step t=1, m starts at 0 and gets updated:
//         m = 0.9 * 0 + 0.1 * grad = 0.1 * grad
//       That 0.1 * grad is way smaller than the actual gradient!
//       Without correction, early steps are tiny and training is slow to start.
//
//       The correction divides by (1 - 0.9^1) = 0.1, giving back: grad
//       At t=2: divide by (1 - 0.9^2) = 0.19 → larger correction
//       At t=100: divide by (1 - 0.9^100) ≈ 1.0 → correction fades away
//
//       In short: bias correction makes the early steps the right size.
//
//  ─────────────────────────────────────────────────────────────────────────────
//
//  THE FULL UPDATE RULE:
//
//    m  = β1 * m  + (1 - β1) * grad          step 1: update momentum
//    v  = β2 * v  + (1 - β2) * grad²         step 2: update variance
//    m̂  = m / (1 - β1^t)                     step 3: bias-correct momentum
//    v̂  = v / (1 - β2^t)                     step 4: bias-correct variance
//    θ  -= lr * m̂ / (√v̂ + ε)               step 5: update parameter
//
//    ε (epsilon) = 1e-8, prevents division by zero when v is tiny
//
//  ─────────────────────────────────────────────────────────────────────────────
//
//  INTUITION for the final step:
//    lr * m̂ / √v̂
//    = lr * (direction + speed) / (how noisy this parameter has been)
//    = effective learning rate that's BIGGER for consistent params,
//      SMALLER for noisy/oscillating params
//
//  Typical hyperparameters: lr=0.001, β1=0.9, β2=0.999, ε=1e-8
//  Notice lr=0.001 vs SGD's lr=0.05 — Adam takes smaller raw steps but
//  adapts them per-parameter, so it's much more efficient overall.
//
// =============================================================================

class Adam {

    /**
     * One Adam instance manages ALL parameters of the model.
     * It stores m (momentum) and v (variance) for every parameter,
     * in the exact same shape as the parameter arrays.
     *
     * @param {number} lr   - learning rate (default 0.001)
     * @param {number} b1   - momentum decay factor (default 0.9)
     * @param {number} b2   - variance decay factor (default 0.999)
     * @param {number} eps  - numerical stability (default 1e-8)
     */
    constructor(lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
        this.lr = lr;
        this.b1 = b1;
        this.b2 = b2;
        this.eps = eps;
        this.t = 0;       // step counter — used for bias correction

        // We lazily create m and v buffers when we first see each parameter.
        // Key = a name string you give, value = { m: ..., v: ... }
        this.state = {};
    }

    // Call once per training sample, BEFORE updating any parameters
    step() { this.t += 1; }

    // Precompute bias correction denominators (same for all params at step t)
    _bc() {
        return {
            bc1: 1 - Math.pow(this.b1, this.t),
            bc2: 1 - Math.pow(this.b2, this.t),
        };
    }

    // Update a 2D weight matrix W using its gradient dW
    //   name = unique string to identify this parameter (e.g. 'W1', 'W2')
    updateMatrix(name, W, dW) {
        if (!this.state[name]) {
            this.state[name] = {
                m: M.zeros(W.length, W[0].length),
                v: M.zeros(W.length, W[0].length)
            };
        }
        const { m, v } = this.state[name];
        const { bc1, bc2 } = this._bc();

        for (let i = 0; i < W.length; i++) {
            for (let j = 0; j < W[i].length; j++) {
                const g = dW[i][j];

                // Step 1 & 2: update running averages
                m[i][j] = this.b1 * m[i][j] + (1 - this.b1) * g;
                v[i][j] = this.b2 * v[i][j] + (1 - this.b2) * g * g;

                // Step 3 & 4: bias correction
                const mHat = m[i][j] / bc1;
                const vHat = v[i][j] / bc2;

                // Step 5: adaptive parameter update
                W[i][j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
            }
        }
    }

    // Update a 1D bias vector b using its gradient db
    updateVector(name, b, db) {
        if (!this.state[name]) {
            this.state[name] = {
                m: new Float32Array(b.length),
                v: new Float32Array(b.length)
            };
        }
        const { m, v } = this.state[name];
        const { bc1, bc2 } = this._bc();

        for (let i = 0; i < b.length; i++) {
            const g = db[i];

            m[i] = this.b1 * m[i] + (1 - this.b1) * g;
            v[i] = this.b2 * v[i] + (1 - this.b2) * g * g;

            const mHat = m[i] / bc1;
            const vHat = v[i] / bc2;

            b[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
        }
    }

    // Sparse embedding update — only update the rows that were actually used.
    //
    //   WHY sparse? If vocab has 50,000 tokens but only 4 appear in this sample,
    //   there's no gradient for the other 49,996 rows. Updating them with zero
    //   gradient would still advance the step counter for their m and v,
    //   making bias correction wrong. So we only touch rows that were used.
    //
    updateEmbedding(name, table, rowIndices, dConcatGrad, embDim) {
        if (!this.state[name]) {
            this.state[name] = {
                m: M.zeros(table.length, embDim),
                v: M.zeros(table.length, embDim)
            };
        }
        const { m, v } = this.state[name];
        const { bc1, bc2 } = this._bc();

        for (let pos = 0; pos < rowIndices.length; pos++) {
            const row = rowIndices[pos];
            for (let d = 0; d < embDim; d++) {
                const g = dConcatGrad[pos * embDim + d];

                m[row][d] = this.b1 * m[row][d] + (1 - this.b1) * g;
                v[row][d] = this.b2 * v[row][d] + (1 - this.b2) * g * g;

                const mHat = m[row][d] / bc1;
                const vHat = v[row][d] / bc2;

                table[row][d] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
            }
        }
    }
}




// =============================================================================
//  PART 6 — MLP LANGUAGE MODEL WITH ADAM
//
//  Now we look at N previous characters to predict the next one.
//  This is the Bengio 2003 "Neural Probabilistic Language Model" paper.
//  It's the direct ancestor of every modern language model.
//
//  Architecture:
//
//    [t-N, ..., t-2, t-1]       ← context window of N tokens
//          ↓ EmbeddingTable
//    [emb_1, emb_2, ..., emb_N] ← each token becomes a dense vector
//          ↓ concatenate
//    [one long vector: N * embDim floats]
//          ↓ W1, b1, tanh
//    [hiddenSize neurons]        ← the hidden layer learns combinations
//          ↓ W2, b2
//    [vocabSize logits]          ← one score per possible next token
//          ↓ softmax
//    [probabilities]             → sample or argmax to get next token
//
//  The difference from Part 4 (bigram): instead of one embedding,
//  we feed N embeddings concatenated. The hidden layer then learns what
//  COMBINATIONS of previous characters predict the next one.
//
//  With Adam replacing plain SGD, this trains 3-5x faster.
// =============================================================================

class MLPLanguageModel {

    /**
     * @param {string[]} vocab       - all unique characters (including <pad>)
     * @param {number} contextSize   - how many previous tokens to look at
     * @param {number} embeddingDim  - size of each token's embedding vector
     * @param {number} hiddenSize    - neurons in the hidden layer
     * @param {number} lr            - Adam learning rate (0.001 is standard)
     */
    constructor(vocab, contextSize = 3, embeddingDim = 10, hiddenSize = 64, lr = 0.001) {
        this.vocab = vocab;
        this.vocabSize = vocab.length;
        this.contextSize = contextSize;
        this.embeddingDim = embeddingDim;
        this.hiddenSize = hiddenSize;
        this.charToIdx = Object.fromEntries(vocab.map((c, i) => [c, i]));

        // Embedding table: each of the vocabSize tokens gets an embeddingDim vector
        this.embedTable = M.rand(this.vocabSize, embeddingDim, 0.1);

        // MLP input size = contextSize tokens × embeddingDim floats each
        const mlpInput = contextSize * embeddingDim;

        // Hidden layer: mlpInput → hiddenSize
        this.W1 = M.rand(hiddenSize, mlpInput);
        this.b1 = new Float32Array(hiddenSize);

        // Output layer: hiddenSize → vocabSize (logits, then softmax)
        this.W2 = M.rand(this.vocabSize, hiddenSize);
        this.b2 = new Float32Array(this.vocabSize);

        // ★ Adam optimizer — one instance manages ALL 4 parameter groups above
        //   plus the embedding table. It will lazily create m/v buffers for each.
        this.adam = new Adam(lr);
    }

    // Embed all context tokens and concatenate into one flat vector
    _embed(contextIndices) {
        const out = new Float32Array(contextIndices.length * this.embeddingDim);
        for (let i = 0; i < contextIndices.length; i++) {
            out.set(this.embedTable[contextIndices[i]], i * this.embeddingDim);
        }
        return out;
    }

    // Forward pass: context indices → probability distribution over next tokens
    forward(contextIndices) {
        this.ctx = contextIndices;
        this.embConcat = this._embed(contextIndices);   // [contextSize * embDim]

        // Hidden layer
        this.z1 = M.addBias(M.matVec(this.W1, this.embConcat), this.b1);
        this.a1 = M.tanh(this.z1);

        // Output logits + probabilities
        this.logits = M.addBias(M.matVec(this.W2, this.a1), this.b2);
        this.probs = M.softmax(this.logits);

        return this.probs;
    }

    // Train on one (context → target) pair
    train(contextIndices, targetIndex) {
        const probs = this.forward(contextIndices);
        const loss = M.crossEntropy(probs, targetIndex);

        // ================================================================
        //  BACKPROPAGATION
        //  Same math as always — compute how much each weight contributed
        //  to the loss, working backwards from output to input.
        // ================================================================

        // Gradient at the output (softmax + cross-entropy simplified)
        const dLogits = M.softmaxCrossEntropyGrad(probs, targetIndex);

        // Layer 2 gradients
        const dW2 = M.zeros(this.vocabSize, this.hiddenSize);
        const db2 = new Float32Array(this.vocabSize);
        const da1 = new Float32Array(this.hiddenSize);

        for (let i = 0; i < this.vocabSize; i++) {
            db2[i] = dLogits[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                dW2[i][j] = dLogits[i] * this.a1[j];
                da1[j] += dLogits[i] * this.W2[i][j];
            }
        }

        // Backprop through tanh: d(tanh(z)) = 1 - tanh(z)^2
        const dz1 = this.z1.map((z, i) => da1[i] * (1 - Math.tanh(z) ** 2));

        // Layer 1 gradients + gradient w.r.t. the concatenated embeddings
        const dW1 = M.zeros(this.hiddenSize, this.embConcat.length);
        const db1 = new Float32Array(this.hiddenSize);
        const dEmbConcat = new Float32Array(this.embConcat.length);

        for (let i = 0; i < this.hiddenSize; i++) {
            db1[i] = dz1[i];
            for (let j = 0; j < this.embConcat.length; j++) {
                dW1[i][j] = dz1[i] * this.embConcat[j];
                dEmbConcat[j] += dz1[i] * this.W1[i][j];
            }
        }

        // ================================================================
        //  ADAM UPDATE
        //  The ONLY change vs plain SGD: instead of `param -= lr * grad`,
        //  we hand the gradient to Adam and it applies the adaptive update.
        //
        //  Compare with plain SGD (from MLPLanguageModel above):
        //    this.W2[i][j] -= this.lr * dW2[i][j];   ← SGD
        //    this.adam.updateMatrix('W2', this.W2, dW2);  ← Adam
        //
        //  Same gradient, smarter step size.
        // ================================================================

        this.adam.step();   // ← increment the global step counter t

        this.adam.updateMatrix('W2', this.W2, dW2);
        this.adam.updateVector('b2', this.b2, db2);
        this.adam.updateMatrix('W1', this.W1, dW1);
        this.adam.updateVector('b1', this.b1, db1);
        this.adam.updateEmbedding(
            'emb', this.embedTable, this.ctx, dEmbConcat, this.embeddingDim
        );

        return loss;
    }

    // Build (context, target) pairs from a raw text string
    //   "hello" with contextSize=2:
    //     ([<pad>, <pad>] → 'h'), ([<pad>, 'h'] → 'e'), (['h','e'] → 'l'), ...
    buildDataset(text) {
        const pad = Array(this.contextSize).fill(0);   // index 0 = <pad>
        const indices = [...pad, ...text.split('').map(c => this.charToIdx[c] ?? 0)];
        const pairs = [];
        for (let i = this.contextSize; i < indices.length; i++) {
            pairs.push({
                ctx: indices.slice(i - this.contextSize, i),
                target: indices[i],
            });
        }
        return pairs;
    }

    // Generate text auto-regressively: predict one token, feed it back in, repeat
    //
    //   temperature controls randomness:
    //     0.5  → more conservative, sticks to common patterns
    //     1.0  → balanced
    //     1.5  → more creative / random
    //
    //   WHY temperature works:
    //     Divide logits by T before softmax.
    //     T < 1: logits get bigger relative to each other → sharper distribution
    //     T > 1: logits get smaller relative to each other → flatter distribution
    //
    generate(seedText, length = 60, temperature = 0.8) {
        // Build starting context from seed, padded if seed is shorter than contextSize
        const pad = Array(this.contextSize).fill(0);
        let ctx = [...pad, ...seedText.split('').map(c => this.charToIdx[c] ?? 0)];
        ctx = ctx.slice(-this.contextSize);

        let out = seedText;

        for (let i = 0; i < length; i++) {
            // Forward pass to get logits
            this.forward(ctx);

            // Apply temperature scaling to logits before softmax
            const scaled = this.logits.map(l => l / temperature);
            const probs = M.softmax(scaled);

            // Sample the next token
            const next = sampleFromProbs(probs);
            out += this.vocab[next];

            // Slide the context window forward by one
            ctx = [...ctx.slice(1), next];
        }
        return out;
    }
}




// =============================================================================
//  HELPER — Sample from a probability distribution
//
//  Instead of always picking the most likely token (greedy = repetitive),
//  we pick randomly proportional to the probabilities.
//
//  If probs = [0.7, 0.2, 0.1]:
//    70% of the time we pick index 0
//    20% of the time we pick index 1
//    10% of the time we pick index 2
//
//  This gives the model "creativity" — it won't always repeat the same phrase.
// =============================================================================

function sampleFromProbs(probs) {
    const r = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probs.length; i++) {
        cumSum += probs[i];
        if (r < cumSum) return i;
    }
    return probs.length - 1;
}




// =============================================================================
//  DEMO — TRAIN AND COMPARE: PLAIN SGD vs ADAM
//
//  We train two identical models on the same data with the same shuffles.
//  The only difference: one uses plain gradient descent, one uses Adam.
//
//  Watch what happens to the loss numbers.
// =============================================================================

function runDemo() {
    console.log("\n" + "=".repeat(70));
    console.log("  MLP LANGUAGE MODEL — PLAIN SGD vs ADAM");
    console.log("=".repeat(70));

    const text = "Hello world. The sun was rising over the quiet town, and the streets were still empty. A small cat sat on the warm stone wall, watching the birds flutter between the trees. The dog ran far away when it heard a sudden noise, but soon returned, curious and cautious. People slowly stepped outside, greeting each other with a soft hello and a tired smile. By noon, the market was busy, full of color, laughter, and the scent of fresh bread. As evening came, the sky turned orange and the world grew calm again. Hello again, whispered the wind as night settled in.";

    // Build vocab from unique characters in the text
    const vocab = ['<pad>', ...new Set(text.split(''))].sort();
    console.log(`\nVocab size: ${vocab.length} characters`);

    // Hyperparameters — SAME for both models
    const CONTEXT = 4;
    const EMB_DIM = 12;
    const HIDDEN = 128;
    const EPOCHS = 5000;

    // ── SGD MODEL (plain gradient descent) ───────────────────────────────────
    //   Uses the old update rule: param -= lr * grad
    //   Needs a higher lr than Adam (0.05 vs 0.001) to have any hope of competing
    class MLPLanguageModelSGD {
        constructor(vocab, contextSize, embeddingDim, hiddenSize, lr) {
            this.vocab = vocab;
            this.vocabSize = vocab.length;
            this.contextSize = contextSize;
            this.embeddingDim = embeddingDim;
            this.hiddenSize = hiddenSize;
            this.lr = lr;
            this.charToIdx = Object.fromEntries(vocab.map((c, i) => [c, i]));
            this.embedTable = M.rand(this.vocabSize, embeddingDim, 0.1);
            const mlpInput = contextSize * embeddingDim;
            this.W1 = M.rand(hiddenSize, mlpInput);
            this.b1 = new Float32Array(hiddenSize);
            this.W2 = M.rand(this.vocabSize, hiddenSize);
            this.b2 = new Float32Array(this.vocabSize);
        }

        _embed(ctx) {
            const out = new Float32Array(ctx.length * this.embeddingDim);
            for (let i = 0; i < ctx.length; i++)
                out.set(this.embedTable[ctx[i]], i * this.embeddingDim);
            return out;
        }

        forward(ctx) {
            this.ctx = ctx;
            this.embConcat = this._embed(ctx);
            this.z1 = M.addBias(M.matVec(this.W1, this.embConcat), this.b1);
            this.a1 = M.tanh(this.z1);
            this.logits = M.addBias(M.matVec(this.W2, this.a1), this.b2);
            this.probs = M.softmax(this.logits);
            return this.probs;
        }

        train(ctx, targetIndex) {
            const probs = this.forward(ctx);
            const loss = M.crossEntropy(probs, targetIndex);
            const dLogits = M.softmaxCrossEntropyGrad(probs, targetIndex);

            const dW2 = M.zeros(this.vocabSize, this.hiddenSize);
            const db2 = new Float32Array(this.vocabSize);
            const da1 = new Float32Array(this.hiddenSize);

            for (let i = 0; i < this.vocabSize; i++) {
                db2[i] = dLogits[i];
                for (let j = 0; j < this.hiddenSize; j++) {
                    dW2[i][j] = dLogits[i] * this.a1[j];
                    da1[j] += dLogits[i] * this.W2[i][j];
                }
            }

            const dz1 = this.z1.map((z, i) => da1[i] * (1 - Math.tanh(z) ** 2));
            const dW1 = M.zeros(this.hiddenSize, this.embConcat.length);
            const db1 = new Float32Array(this.hiddenSize);
            const dEmbConcat = new Float32Array(this.embConcat.length);

            for (let i = 0; i < this.hiddenSize; i++) {
                db1[i] = dz1[i];
                for (let j = 0; j < this.embConcat.length; j++) {
                    dW1[i][j] = dz1[i] * this.embConcat[j];
                    dEmbConcat[j] += dz1[i] * this.W1[i][j];
                }
            }

            // ── PLAIN SGD UPDATE ─────────────────────────────────────────────
            //   Simple, readable, but suboptimal.
            for (let i = 0; i < this.vocabSize; i++) {
                this.b2[i] -= this.lr * db2[i];
                for (let j = 0; j < this.hiddenSize; j++)
                    this.W2[i][j] -= this.lr * dW2[i][j];
            }
            for (let i = 0; i < this.hiddenSize; i++) {
                this.b1[i] -= this.lr * db1[i];
                for (let j = 0; j < this.embConcat.length; j++)
                    this.W1[i][j] -= this.lr * dW1[i][j];
            }
            for (let pos = 0; pos < this.ctx.length; pos++) {
                const row = this.embedTable[this.ctx[pos]];
                for (let d = 0; d < this.embeddingDim; d++)
                    row[d] -= this.lr * dEmbConcat[pos * this.embeddingDim + d];
            }

            return loss;
        }

        buildDataset(text) {
            const pad = Array(this.contextSize).fill(0);
            const indices = [...pad, ...text.split('').map(c => this.charToIdx[c] ?? 0)];
            const pairs = [];
            for (let i = this.contextSize; i < indices.length; i++)
                pairs.push({ ctx: indices.slice(i - this.contextSize, i), target: indices[i] });
            return pairs;
        }
    }

    // Create both models — identical architecture, different optimizers
    const sgdModel = new MLPLanguageModelSGD(vocab, CONTEXT, EMB_DIM, HIDDEN, 0.05);
    const adamModel = new MLPLanguageModel(vocab, CONTEXT, EMB_DIM, HIDDEN, 0.001);

    // Build the same dataset for both
    const dataset = sgdModel.buildDataset(text);
    console.log(`Training pairs: ${dataset.length}, Epochs: ${EPOCHS}\n`);

    const LOG_EVERY = 30;
    console.log(`${'Epoch'.padEnd(8)} ${'SGD Loss'.padEnd(14)} ${'Adam Loss'.padEnd(14)} Winner`);
    console.log('─'.repeat(50));

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        // SAME shuffle for both — fair comparison
        const shuffled = dataset.slice().sort(() => Math.random() - 0.5);

        let sgdLoss = 0, adamLoss = 0;
        shuffled.forEach(({ ctx, target }) => {
            sgdLoss += sgdModel.train(ctx, target);
            adamLoss += adamModel.train(ctx, target);
        });

        if (epoch % LOG_EVERY === 0 || epoch === EPOCHS - 1) {
            const avgSgd = (sgdLoss / dataset.length).toFixed(4);
            const avgAdam = (adamLoss / dataset.length).toFixed(4);
            const winner = parseFloat(avgAdam) < parseFloat(avgSgd) ? '← Adam' : '← SGD ';
            console.log(`${String(epoch).padEnd(8)} ${avgSgd.padEnd(14)} ${avgAdam.padEnd(14)} ${winner}`);
        }
    }

    // Generate text with the Adam model
    console.log("\n--- Adam model (temp=0.8) ---");
    const seeds = ['Hel', 'The', 'the', 'A s'];
    seeds.forEach(seed => {
        console.log(`  seed="${seed}" → "${adamModel.generate(seed, 155, 0.8)}"`);
    });

    console.log("\n--- Same seeds, temp=0.4 (more conservative) ---");
    seeds.forEach(seed => {
        console.log(`  seed="${seed}" → "${adamModel.generate(seed, 155, 0.4)}"`);
    });

    console.log("\n--- Same seeds, temp=1.4 (more random) ---");
    seeds.forEach(seed => {
        console.log(`  seed="${seed}" → "${adamModel.generate(seed, 155, 1.4)}"`);
    });
}

runDemo();




// =============================================================================
//  PART 7 — WHAT'S NEXT: THE ROAD TO TRANSFORMERS
//
//  You now have: embeddings, softmax, cross-entropy, backprop, Adam.
//  That's the complete toolkit for a real 2003-era language model.
//
//  Here's what separates this from GPT and WHY each piece exists:
//
//
//  ① SINGLE ATTENTION HEAD  ← your next implementation target
//  ─────────────────────────
//  Your MLP concatenates N embeddings and mashes them through a hidden layer.
//  That means position 1 can only interact with position 3 through the weights.
//
//  Attention lets every token directly "look at" every other token:
//
//    Q[i] = W_Q · token[i]    // Query:  "what am I looking for?"
//    K[j] = W_K · token[j]    // Key:    "what do I contain?"
//    V[j] = W_V · token[j]    // Value:  "what information do I give?"
//
//    score[i][j] = dot(Q[i], K[j]) / sqrt(headDim)   // how relevant is j to i?
//    weight[i]   = softmax(score[i])                  // normalize to probabilities
//    output[i]   = sum_j(weight[i][j] * V[j])        // weighted mix of values
//
//  The division by sqrt(headDim) prevents scores from getting too large
//  (which would make softmax outputs very close to 0 or 1, killing gradients).
//
//  The CAUSAL MASK: set score[i][j] = -Infinity for j > i.
//  This means position i cannot attend to future positions.
//  After softmax, -Infinity → 0 weight. The future is invisible.
//
//
//  ② MULTI-HEAD ATTENTION
//  ─────────────────────────
//  Run H independent attention heads in parallel, each with smaller dimensions.
//    headDim = embDim / numHeads
//  Concatenate all outputs, then project back to embDim.
//
//  WHY? Each head learns to attend to different types of relationships.
//  One head might track subject-verb agreement, another tracks coreference.
//  They specialize automatically during training.
//
//
//  ③ RESIDUAL CONNECTIONS
//  ─────────────────────────
//  x = x + Attention(LayerNorm(x))
//  x = x + FFN(LayerNorm(x))
//
//  The "+x" is a skip connection. Without it, gradients vanish through 12+ layers.
//  With it, a gradient can flow directly from output to input in one step.
//  This is why transformers can be 96 layers deep when MLPs can only go ~5.
//
//
//  ④ LAYER NORMALIZATION
//  ─────────────────────────
//  For each token's vector x of length d:
//    μ = mean(x)
//    σ = std(x)
//    x_norm = (x - μ) / (σ + ε)        // normalize to mean=0, std=1
//    output  = gamma * x_norm + beta    // learned scale and shift
//
//  WHY? Keeps activations in a stable range throughout a deep network.
//  Without it, values drift and training becomes unstable after ~4 layers.
//  gamma and beta are learned parameters (also updated by Adam!).
//
//
//  ⑤ POSITIONAL ENCODING
//  ─────────────────────────
//  Attention has no concept of order — [a, b, c] and [c, b, a] look identical.
//  You must tell the model WHERE each token is.
//
//  Simplest approach (GPT-2): a second embedding table, indexed by position.
//    input[pos] = tokenEmbed[token[pos]] + posEmbed[pos]
//  posEmbed is just a [maxSeqLen x embDim] matrix, learned during training.
//
//
//  ⑥ FEED-FORWARD SUBLAYER
//  ─────────────────────────
//  After attention, each token is processed independently through a small MLP:
//    FFN(x) = GELU(x · W1 + b1) · W2 + b2
//    hidden size = 4 × embDim (GPT convention)
//
//  This is where the model "thinks" about what it just attended to.
//  Attention gathers information; FFN processes it.
//
//  GELU is a smoother alternative to ReLU used in modern transformers:
//    GELU(x) ≈ x * sigmoid(1.702 * x)
//
//
//  ⑦ ONE FULL TRANSFORMER BLOCK:
//  ─────────────────────────────────────────────────
//    x = x + MultiHeadAttention(LayerNorm(x))
//    x = x + FFN(LayerNorm(x))
//
//  GPT-2 small: embDim=768, 12 heads, 12 blocks, 117M parameters
//  GPT-3:       embDim=12288, 96 heads, 96 blocks, 175B parameters
//  Same structure, 1500x more parameters. Trained on 100x more data.
//
//
//  ⑧ YOUR ROADMAP FROM HERE:
//  ─────────────────────────────────────────────────
//
//  WEEK 1: Implement CausalSelfAttention (forward pass only, no backprop yet)
//    → W_Q, W_K, W_V matrices
//    → compute scores, mask, softmax, weighted sum
//    → make sure shapes are correct
//
//  WEEK 2: Add backprop through attention
//    → dV, dK, dQ from dOutput
//    → dScore from dWeight (through softmax)
//    → chain rule back to W_Q, W_K, W_V and the input tokens
//
//  WEEK 3: Add LayerNorm + residual connections
//    → LN forward: normalize, scale, shift
//    → LN backward: chain rule through mean and std (tricky but doable)
//    → Residual: just add the input back to the output, gradient flows both ways
//
//  WEEK 4: Stack N transformer blocks, add positional embeddings
//    → This is just a loop: for each block, apply attention sublayer then FFN sublayer
//    → Train on something larger (e.g., a book chapter or Wikipedia article)
//
//  You will have a working baby GPT. Not metaphorically. Actually.
//
// =============================================================================