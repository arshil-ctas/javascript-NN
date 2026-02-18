// =============================================================================
//  FROM PERCEPTRON TO LANGUAGE MODEL — Pure JavaScript, No Libraries
//  A learning journey: your MLP → Embeddings → Softmax → Next-Token Prediction
// =============================================================================
//
//  WHAT THIS FILE TEACHES (in order):
//
//  PART 0 — Math Primitives (matrix ops you'll need everywhere)
//  PART 1 — Your Original MLP, generalized & cleaned
//  PART 2 — Softmax + Cross-Entropy (how to predict from a vocabulary)
//  PART 3 — Embeddings (turning tokens into learnable vectors)
//  PART 4 — Bigram Language Model (simplest possible next-token predictor)
//  PART 5 — MLP Language Model (context window → next token, like early GPT)
//  PART 6 — Conceptual Map: What's left to become a real Transformer
//
// =============================================================================




// =============================================================================
//  PART 0 — MATH PRIMITIVES
//  Real neural networks are just matrix multiplications + nonlinearities.
//  You need these helpers everywhere.
// =============================================================================

const M = {

    // Create a 2D matrix of zeros: shape [rows x cols]
    zeros: (rows, cols) =>
        Array.from({ length: rows }, () => new Float32Array(cols)),

    // Random matrix, Xavier-ish init: helps gradients not vanish/explode
    rand: (rows, cols, scale) => {
        scale = scale ?? Math.sqrt(2.0 / (rows + cols)); // Xavier init
        return Array.from({ length: rows }, () =>
            Float32Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
        );
    },

    // Random vector
    randVec: (n, scale) => {
        scale = scale ?? 0.1;
        return Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * scale);
    },

    // Matrix × vector: [rows x cols] · [cols] → [rows]
    //   This is the core of every dense layer
    matVec: (W, x) => {
        const out = new Float32Array(W.length);
        for (let i = 0; i < W.length; i++) {
            let s = 0;
            for (let j = 0; j < x.length; j++) s += W[i][j] * x[j];
            out[i] = s;
        }
        return out;
    },

    // Vector dot product
    dot: (a, b) => {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    },

    // Add bias vector to result in-place
    addBias: (z, b) => {
        for (let i = 0; i < z.length; i++) z[i] += b[i];
        return z;
    },

    // Element-wise ReLU
    relu: (z) => z.map(v => Math.max(0, v)),

    // ReLU derivative (for backprop)
    reluGrad: (z) => z.map(v => v > 0 ? 1 : 0),

    // Tanh — smoother than ReLU, good for small networks
    tanh: (z) => z.map(Math.tanh),

    // Softmax: turns raw scores (logits) into a probability distribution
    //   CRITICAL: we subtract max first for numerical stability (avoids e^huge)
    softmax: (logits) => {
        const maxL = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxL));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    },

    // Cross-entropy loss: -log(probability of the correct class)
    //   This is THE loss function for classification / language modeling
    crossEntropy: (probs, targetIndex) => -Math.log(probs[targetIndex] + 1e-9),

    // Gradient of cross-entropy + softmax combined (they simplify beautifully):
    //   dLoss/dLogit[i] = probs[i] - (1 if i==target else 0)
    softmaxCrossEntropyGrad: (probs, targetIndex) => {
        const grad = Float32Array.from(probs);
        grad[targetIndex] -= 1.0;
        return grad;
    },
};




// =============================================================================
//  PART 1 — YOUR MLP, REWRITTEN WITH THE MATH PRIMITIVES
//  Same network, cleaner code, ready to be extended.
// =============================================================================

class MLP {

    /**
     * N inputs → H hidden → vocabSize outputs (with softmax)
     * This version predicts a CLASS, not just 0/1.
     */
    constructor(inputSize, hiddenSize, outputSize, lr = 0.05) {
        this.lr = lr;

        // Layer 1: input → hidden
        this.W1 = M.rand(hiddenSize, inputSize);
        this.b1 = new Float32Array(hiddenSize); // zeros

        // Layer 2: hidden → output (logits, before softmax)
        this.W2 = M.rand(outputSize, hiddenSize);
        this.b2 = new Float32Array(outputSize);
    }

    forward(x) {
        // Hidden layer
        this.x = x;
        this.z1 = M.addBias(M.matVec(this.W1, x), this.b1);
        this.a1 = M.tanh(this.z1);                     // activation

        // Output logits
        this.logits = M.addBias(M.matVec(this.W2, this.a1), this.b2);
        this.probs = M.softmax(this.logits);           // probabilities

        return this.probs;
    }

    // targetIndex = the index of the correct answer in the vocabulary
    train(x, targetIndex) {
        const probs = this.forward(x);
        const loss = M.crossEntropy(probs, targetIndex);

        // --- Backprop ---

        // Gradient from softmax+crossentropy (the math works out to this)
        const dLogits = M.softmaxCrossEntropyGrad(probs, targetIndex);

        // Layer 2 gradients
        const dW2 = M.zeros(this.W2.length, this.a1.length);
        const db2 = new Float32Array(this.b2.length);
        const da1 = new Float32Array(this.a1.length);

        for (let i = 0; i < this.W2.length; i++) {
            db2[i] = dLogits[i];
            for (let j = 0; j < this.a1.length; j++) {
                dW2[i][j] = dLogits[i] * this.a1[j];
                da1[j] += dLogits[i] * this.W2[i][j]; // accumulate
            }
        }

        // Through tanh: derivative of tanh(z) = 1 - tanh(z)^2
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

        // --- Update (gradient descent) ---
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
//  PART 2 — WHY SOFTMAX + CROSS-ENTROPY?
//  Your original network predicted 0 or 1 (binary).
//  Language models predict from a VOCABULARY (could be 50,000 tokens).
//
//  The softmax turns raw scores (logits) into probabilities that sum to 1.
//  Cross-entropy punishes the model when it gives low probability to the right answer.
//
//  Example: vocab = ['a','b','c','d']
//   logits = [2.0, 0.5, -1.0, 0.1]
//   softmax → [0.63, 0.14, 0.026, 0.20]
//   if correct answer is 'a' (index 0): loss = -log(0.63) ≈ 0.46  ✓ low loss
//   if correct answer is 'b' (index 1): loss = -log(0.14) ≈ 1.97  ✗ high loss
// =============================================================================

function demonstrateSoftmax() {
    console.log("\n=== PART 2: Softmax Demo ===");
    const logits = [2.0, 0.5, -1.0, 0.1];
    const probs = M.softmax(logits);
    console.log("Logits:", Array.from(logits));
    console.log("Probs: ", Array.from(probs).map(p => p.toFixed(3)));
    console.log("Sum:   ", probs.reduce((a, b) => a + b, 0).toFixed(6), "(always 1.0)");
    console.log("Loss if target=0:", M.crossEntropy(probs, 0).toFixed(4), "(low = good)");
    console.log("Loss if target=1:", M.crossEntropy(probs, 1).toFixed(4), "(high = bad)");
}
demonstrateSoftmax();




// =============================================================================
//  PART 3 — EMBEDDINGS
//  This is where things get interesting.
//
//  Problem: you can't feed a character like 'a' directly to a neural network.
//           You need numbers. You COULD use one-hot encoding [0,0,1,0,0,...],
//           but that's wasteful (vocabSize dimensions, all zeros except one).
//
//  Solution: Embedding table.
//   - A matrix of shape [vocabSize x embeddingDim]
//   - Each token gets its own row — a dense vector of learned numbers
//   - Looking up token 3 = just grabbing row 3 of the matrix
//   - Those vectors LEARN to encode meaning (e.g. similar chars end up close)
//
//  This is the exact same thing as word2vec / GPT token embeddings.
// =============================================================================

class EmbeddingTable {

    /**
     * @param {number} vocabSize    - how many unique tokens
     * @param {number} embeddingDim - size of each token's vector
     */
    constructor(vocabSize, embeddingDim, lr = 0.05) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.lr = lr;

        // The embedding matrix: [vocabSize x embeddingDim]
        // Each row is one token's vector, initialized randomly (small values)
        this.table = M.rand(vocabSize, embeddingDim, 0.1);
    }

    // "Lookup" — just grab the row for this token index
    lookup(tokenIndex) {
        return this.table[tokenIndex]; // returns a Float32Array of length embeddingDim
    }

    // For a context window of multiple tokens, concatenate their embeddings
    // e.g. context [3, 7, 2] → embed each → concat → one long vector
    lookupAndConcat(tokenIndices) {
        const result = new Float32Array(tokenIndices.length * this.embeddingDim);
        for (let i = 0; i < tokenIndices.length; i++) {
            result.set(this.table[tokenIndices[i]], i * this.embeddingDim);
        }
        return result;
    }

    // Gradient update: the gradient flows BACK into the embedding vectors
    // Only the rows that were actually used get updated (sparse update)
    updateGrad(tokenIndices, grad, embeddingDim) {
        // grad is the gradient w.r.t. the concatenated embedding vector
        for (let i = 0; i < tokenIndices.length; i++) {
            const row = this.table[tokenIndices[i]];
            for (let d = 0; d < embeddingDim; d++) {
                row[d] -= this.lr * grad[i * embeddingDim + d];
            }
        }
    }
}




// =============================================================================
//  PART 4 — BIGRAM LANGUAGE MODEL
//  The simplest possible "predict next token" model.
//
//  Input:  one character (e.g. 'h')
//  Output: probability distribution over next character (what comes after 'h'?)
//
//  Architecture: Embedding(1 token) → Linear → Softmax
//  No hidden layer needed — this is essentially a lookup table
//  that learns "given char X, what char is likely next?"
//
//  This is exactly what Andrej Karpathy's makemore starts with.
// =============================================================================

class BigramLM {

    constructor(vocab, embeddingDim = 8, lr = 0.05) {
        this.vocab = vocab;                       // array of unique chars
        this.vocabSize = vocab.length;
        this.charToIdx = Object.fromEntries(vocab.map((c, i) => [c, i]));
        this.lr = lr;

        // Embedding: each char → dense vector
        this.embed = new EmbeddingTable(this.vocabSize, embeddingDim, lr);

        // Output projection: embeddingDim → vocabSize (logits)
        this.W = M.rand(this.vocabSize, embeddingDim);
        this.b = new Float32Array(this.vocabSize);
    }

    forward(prevCharIdx) {
        this.prevIdx = prevCharIdx;
        this.emb = this.embed.lookup(prevCharIdx);          // [embeddingDim]
        this.logits = M.addBias(M.matVec(this.W, this.emb), this.b); // [vocabSize]
        this.probs = M.softmax(this.logits);
        return this.probs;
    }

    train(prevCharIdx, nextCharIdx) {
        const probs = this.forward(prevCharIdx);
        const loss = M.crossEntropy(probs, nextCharIdx);

        // Backprop through softmax+crossentropy
        const dLogits = M.softmaxCrossEntropyGrad(probs, nextCharIdx);

        // Gradient w.r.t. embedding vector
        const dEmb = new Float32Array(this.emb.length);
        for (let i = 0; i < this.W.length; i++) {
            this.b[i] -= this.lr * dLogits[i];
            for (let j = 0; j < this.emb.length; j++) {
                this.W[i][j] -= this.lr * dLogits[i] * this.emb[j];
                dEmb[j] += dLogits[i] * this.W[i][j];
            }
        }

        // Update the embedding for prevCharIdx
        const row = this.embed.table[this.prevIdx];
        for (let d = 0; d < row.length; d++) {
            row[d] -= this.lr * dEmb[d];
        }

        return loss;
    }

    // Sample: given a starting char, generate `length` chars
    generate(startChar, length = 20) {
        let idx = this.charToIdx[startChar] ?? 0;
        let out = startChar;
        for (let i = 0; i < length; i++) {
            const probs = this.forward(idx);
            idx = sampleFromProbs(probs);               // stochastic sampling
            out += this.vocab[idx];
        }
        return out;
    }
}




// =============================================================================
//  PART 5 — MLP LANGUAGE MODEL (Context Window)
//  Now we look at N previous characters to predict the next one.
//  This is the Bengio 2003 paper — the architecture that led to everything.
//
//  Architecture:
//    [t-3, t-2, t-1]  ← context window of 3 tokens
//         ↓ embedding lookup (each token → embDim vector)
//    [emb, emb, emb]  ← concatenated: contextSize * embDim values
//         ↓ W1, b1, tanh
//    [hidden neurons]
//         ↓ W2, b2
//    [logits: vocabSize]
//         ↓ softmax
//    [probabilities]   → pick next token
//
//  The key insight: the model learns to USE the context window.
//  Characters that appear in similar contexts get similar embeddings.
// =============================================================================

class MLPLanguageModel {

    /**
     * @param {string[]} vocab       - unique characters
     * @param {number} contextSize   - how many previous tokens to look at
     * @param {number} embeddingDim  - dimension of each token embedding
     * @param {number} hiddenSize    - neurons in the hidden layer
     */
    constructor(vocab, contextSize = 3, embeddingDim = 10, hiddenSize = 64, lr = 0.05) {
        this.vocab = vocab;
        this.vocabSize = vocab.length;
        this.contextSize = contextSize;
        this.embeddingDim = embeddingDim;
        this.hiddenSize = hiddenSize;
        this.lr = lr;
        this.charToIdx = Object.fromEntries(vocab.map((c, i) => [c, i]));

        // Embedding table: [vocabSize x embeddingDim]
        this.embed = new EmbeddingTable(this.vocabSize, embeddingDim, lr);

        // The input to the MLP = contextSize embeddings concatenated
        const mlpInputSize = contextSize * embeddingDim;

        // Hidden layer
        this.W1 = M.rand(hiddenSize, mlpInputSize);
        this.b1 = new Float32Array(hiddenSize);

        // Output layer
        this.W2 = M.rand(this.vocabSize, hiddenSize);
        this.b2 = new Float32Array(this.vocabSize);
    }

    forward(contextIndices) {
        // 1) Embed + concatenate context tokens
        this.contextIndices = contextIndices;
        this.embConcat = this.embed.lookupAndConcat(contextIndices);

        // 2) Hidden layer
        this.z1 = M.addBias(M.matVec(this.W1, this.embConcat), this.b1);
        this.a1 = M.tanh(this.z1);

        // 3) Output logits
        this.logits = M.addBias(M.matVec(this.W2, this.a1), this.b2);
        this.probs = M.softmax(this.logits);

        return this.probs;
    }

    train(contextIndices, targetIndex) {
        const probs = this.forward(contextIndices);
        const loss = M.crossEntropy(probs, targetIndex);

        // --- Backprop ---

        // dLoss/dLogits (softmax + cross-entropy combined gradient)
        const dLogits = M.softmaxCrossEntropyGrad(probs, targetIndex);

        // Layer 2: dL/dW2, dL/db2, dL/da1
        const da1 = new Float32Array(this.hiddenSize);
        for (let i = 0; i < this.vocabSize; i++) {
            this.b2[i] -= this.lr * dLogits[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                this.W2[i][j] -= this.lr * dLogits[i] * this.a1[j];
                da1[j] += dLogits[i] * this.W2[i][j];
            }
        }

        // Through tanh: dL/dz1
        const dz1 = this.z1.map((z, i) => da1[i] * (1 - Math.tanh(z) ** 2));

        // Layer 1: dL/dW1, dL/db1, dL/dEmbConcat
        const dEmbConcat = new Float32Array(this.embConcat.length);
        for (let i = 0; i < this.hiddenSize; i++) {
            this.b1[i] -= this.lr * dz1[i];
            for (let j = 0; j < this.embConcat.length; j++) {
                this.W1[i][j] -= this.lr * dz1[i] * this.embConcat[j];
                dEmbConcat[j] += dz1[i] * this.W1[i][j];
            }
        }

        // Update embedding table for each context token
        this.embed.updateGrad(contextIndices, dEmbConcat, this.embeddingDim);

        return loss;
    }

    // Build training pairs from a string: sliding window
    // "hello" with context=2 → ([h,e]→l), ([e,l]→l), ([l,l]→o)
    buildDataset(text) {
        const pad = Array(this.contextSize).fill(0); // index 0 = padding/start token
        const indices = [...pad, ...text.split('').map(c => this.charToIdx[c] ?? 0)];
        const pairs = [];
        for (let i = this.contextSize; i < indices.length; i++) {
            pairs.push({
                ctx: indices.slice(i - this.contextSize, i),
                target: indices[i]
            });
        }
        return pairs;
    }

    // Generate text by repeatedly predicting next token
    generate(seedText, length = 50) {
        const pad = Array(this.contextSize).fill(0);
        let ctx = [...pad, ...seedText.split('').map(c => this.charToIdx[c] ?? 0)];
        ctx = ctx.slice(-this.contextSize); // take last N
        let out = seedText;

        for (let i = 0; i < length; i++) {
            const probs = this.forward(ctx);
            const next = sampleFromProbs(probs);
            out += this.vocab[next];
            ctx = [...ctx.slice(1), next]; // slide window
        }
        return out;
    }
}




// =============================================================================
//  HELPER: Sample from a probability distribution
//  Instead of always picking the highest probability (greedy, repetitive),
//  we sample proportionally. This gives variety and creativity.
//
//  This is "temperature" sampling — you could divide logits by T before
//  softmax: T<1 = more confident/repetitive, T>1 = more random/creative
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
//  PART 6 — CONCEPTUAL MAP: WHAT'S NEXT (TOWARD TRANSFORMERS)
//
//  You now have the complete MLP language model. Here's what's missing
//  between this and GPT, and WHY each piece exists:
//
//
//  ① ATTENTION MECHANISM
//  ──────────────────────
//  Problem with your MLP: the context window is FIXED (3 tokens).
//  And all positions are treated independently — there's no way for
//  position 1 to "ask" position 3 a question.
//
//  Attention: each token computes a Query (what am I looking for?),
//  and every other token has a Key (what do I contain?) and Value (what I return).
//  Attention score = softmax(Q · Kᵀ / √d) — this is just a learned weighted average.
//
//  In code (conceptual):
//
//    // For each token position i:
//    Q[i] = W_Q · token[i]      // "what am I looking for?"
//    K[j] = W_K · token[j]      // "what does each token offer?"
//    V[j] = W_V · token[j]      // "what does each token give me?"
//
//    scores[i][j] = dot(Q[i], K[j]) / sqrt(embDim)
//    weights[i]   = softmax(scores[i])              // which tokens to attend to
//    output[i]    = sum_j(weights[i][j] * V[j])    // weighted mixture
//
//  The CAUSAL MASK ensures position i can only look at positions ≤ i
//  (you can't cheat by looking at future tokens during training).
//
//
//  ② MULTI-HEAD ATTENTION
//  ───────────────────────
//  Instead of one Q/K/V, you run H parallel attention "heads" with
//  smaller dimensions, then concatenate. Each head can learn to attend
//  to different relationships (syntax vs. semantics vs. position).
//
//
//  ③ RESIDUAL CONNECTIONS
//  ───────────────────────
//  output = LayerNorm(x + Attention(x))
//  The "+x" (skip connection) lets gradients flow directly back through
//  many layers without vanishing. This is why transformers can be 96 layers deep.
//
//
//  ④ LAYER NORMALIZATION
//  ──────────────────────
//  Normalizes each token's activation vector to mean=0, std=1.
//  Stabilizes training dramatically. Apply before (Pre-LN) or after (Post-LN) each sublayer.
//
//    LN(x) = gamma * (x - mean(x)) / std(x) + beta
//    gamma, beta are learned parameters
//
//
//  ⑤ POSITIONAL ENCODING
//  ──────────────────────
//  Attention is permutation-invariant — it doesn't know position 1 from position 5.
//  You add positional information to each embedding:
//    - Sinusoidal (original GPT): fixed, not learned
//    - Learned positional embeddings (GPT-2): just another embedding table
//
//    embedding[pos] = tokenEmbed[token] + posEmbed[pos]
//
//
//  ⑥ FEED-FORWARD SUBLAYER
//  ────────────────────────
//  After attention, each token goes through a small 2-layer MLP independently:
//    FFN(x) = ReLU(xW1 + b1)W2 + b2
//  with hidden size = 4 × embeddingDim typically.
//  This is where the model "processes" what it attended to.
//
//
//  ⑦ THE FULL TRANSFORMER BLOCK (one layer of GPT):
//  ─────────────────────────────────────────────────
//    x = x + MultiHeadAttention(LayerNorm(x))   // attend & add
//    x = x + FFN(LayerNorm(x))                  // process & add
//
//  GPT-2 small = 12 of these blocks stacked.
//  GPT-3 = 96 blocks. Same idea, just bigger.
//
//
//  ⑧ WHAT MAKES SCALE WORK
//  ────────────────────────
//  - More parameters (bigger embeddings, more heads, more layers)
//  - More data (billions of tokens, not hundreds)
//  - Adam optimizer (adaptive learning rates per parameter — much better than SGD)
//  - Gradient clipping (prevents exploding gradients)
//  - Learning rate schedules (warmup then cosine decay)
//
//  Your next implementation milestone: add Adam optimizer to this MLP LM.
//  Adam update rule (per parameter θ):
//    m = β1*m + (1-β1)*grad          // momentum
//    v = β2*v + (1-β2)*grad²         // RMS
//    θ -= lr * m / (√v + ε)          // parameter update
//
// =============================================================================




// =============================================================================
//  DEMO: TRAIN THE MLP LANGUAGE MODEL
//  Training on a tiny text to predict next character.
//  Watch the loss drop — the model is learning sequence patterns.
// =============================================================================

function runDemo() {
    console.log("\n" + "=".repeat(70));
    console.log("  MLP LANGUAGE MODEL DEMO");
    console.log("=".repeat(70));

    // Tiny training corpus
    const text = "Hello world. The sun was rising over the quiet town, and the streets were still empty. A small cat sat on the warm stone wall, watching the birds flutter between the trees. The dog ran far away when it heard a sudden noise, but soon returned, curious and cautious. People slowly stepped outside, greeting each other with a soft hello and a tired smile. By noon, the market was busy, full of color, laughter, and the scent of fresh bread. As evening came, the sky turned orange and the world grew calm again. Hello again, whispered the wind as night settled in.";
    // Build vocabulary from unique characters
    const vocab = ['<pad>', ...new Set(text.split(''))].sort();
    console.log(`\nVocab size: ${vocab.length} characters`);
    console.log(`Vocab: ${vocab.join(' ')}`);

    // Create the model
    const model = new MLPLanguageModel(
        vocab,
        3,   // context window: look at 3 previous chars
        10,  // embedding dimension
        64,  // hidden neurons
        0.001 // learning rate
    );

    // Build training dataset
    const dataset = model.buildDataset(text);
    console.log(`\nTraining pairs: ${dataset.length}`);

    // Training loop
    const EPOCHS = 100_000;
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        let totalLoss = 0;

        // Shuffle dataset each epoch (important for generalization)
        const shuffled = dataset.slice().sort(() => Math.random() - 0.5);

        shuffled.forEach(({ ctx, target }) => {
            totalLoss += model.train(ctx, target);
        });

        if (epoch % 100 === 0 || epoch === EPOCHS - 1) {
            console.log(`Epoch ${String(epoch).padStart(4)}: loss = ${(totalLoss / dataset.length).toFixed(4)}`);
        }
    }

    // Generate some text
    console.log("\n--- Generated text (seeded with 'hel') ---");
    for (let i = 0; i < 10; i++) {
        console.log(`  "${model.generate('hel', 40)}"`);
    }

    console.log("\n--- Generated text (seeded with 'the') ---");
    for (let i = 0; i < 10; i++) {
        console.log(`  "${model.generate('the', 40)}"`);
    }
}

runDemo();




// =============================================================================
//  YOUR LEARNING ROADMAP FROM HERE
//
//  You're now at the level of a real 2003 NLP paper. Here's what to build next:
//
//  NEXT STEP (2-3 weeks): Add Adam optimizer
//    → Replace the `lr * gradient` update with Adam
//    → You'll see loss drop 3-5x faster
//
//  THEN: Add a second MLP layer (go deeper)
//    → Input → Embed → Hidden1 → Hidden2 → Output
//    → Understand why depth helps (compositionality)
//
//  THEN: Implement a single attention head
//    → Add Q, K, V matrices to your model
//    → Replace the fixed context concat with attention-weighted sum
//    → This is the hardest step — give it a week
//
//  THEN: Multi-head + residuals
//    → Add the skip connections (just `+` the original input)
//    → Add LayerNorm (normalize each row to mean=0 std=1)
//
//  THEN: You have a baby transformer. Train it on something bigger.
//
//  The gap between this code and GPT-2 is:
//    - Adam optimizer
//    - Attention + positional encodings
//    - Stacked transformer blocks (N × the same structure)
//    - Much more data + parameters
//    - That's it. No magic.
//
// =============================================================================