/**
 * Generalized Neural Network
 * N inputs → H hidden neurons → 1 output
 */
class NeuralNetwork {

    /**
     * @param {number} inputSize  - number of input features
     * @param {number} hiddenSize - number of hidden neurons
     */
    constructor(inputSize, hiddenSize) {

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Initialize W1 (hiddenSize x inputSize)
        this.W1 = Array.from({ length: hiddenSize }, () =>
            Array.from({ length: inputSize }, () => Math.random() * 2 - 1)
        );

        // Hidden bias
        this.b1 = Array(hiddenSize).fill(0);

        // Output weights (hiddenSize → 1)
        this.W2 = Array.from({ length: hiddenSize }, () => Math.random() * 2 - 1);

        this.b2 = 0;

        this.learningRate = 0.1;
    }

    relu(x) { return Math.max(0, x); }
    reluDerivative(x) { return x > 0 ? 1 : 0; }

    sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    sigmoidDerivative(x) {
        const s = this.sigmoid(x);
        return s * (1 - s);
    }

    /**
     * Forward pass
     * @param {number[]} x
     */
    forward(x) {

        // ---- Hidden Layer ----
        this.z1 = [];

        for (let i = 0; i < this.hiddenSize; i++) {

            let sum = 0;

            for (let j = 0; j < this.inputSize; j++) {
                sum += this.W1[i][j] * x[j];
            }

            this.z1[i] = sum + this.b1[i];
        }

        this.a1 = this.z1.map(z => this.relu(z));

        // ---- Output ----
        let sum = 0;

        for (let i = 0; i < this.hiddenSize; i++) {
            sum += this.W2[i] * this.a1[i];
        }

        this.z2 = sum + this.b2;
        this.a2 = this.sigmoid(this.z2);

        return this.a2;
    }

    /**
     * Train on one sample
     */
    train(x, y) {

        const output = this.forward(x);

        const loss = Math.pow(output - y, 2);

        const lossGradient = 2 * (output - y);
        const dz2 = lossGradient * this.sigmoidDerivative(this.z2);

        // ---- Gradients for output ----
        const dW2 = this.a1.map(a => dz2 * a);
        const db2 = dz2;

        // ---- Backprop to hidden ----
        const dz1 = [];

        for (let i = 0; i < this.hiddenSize; i++) {
            dz1[i] = dz2 * this.W2[i] * this.reluDerivative(this.z1[i]);
        }

        // ---- Gradients for W1 ----
        const dW1 = [];

        for (let i = 0; i < this.hiddenSize; i++) {
            dW1[i] = [];
            for (let j = 0; j < this.inputSize; j++) {
                dW1[i][j] = dz1[i] * x[j];
            }
        }

        const db1 = dz1;

        // ---- Update ----
        for (let i = 0; i < this.hiddenSize; i++) {

            this.W2[i] -= this.learningRate * dW2[i];
            this.b1[i] -= this.learningRate * db1[i];

            for (let j = 0; j < this.inputSize; j++) {
                this.W1[i][j] -= this.learningRate * dW1[i][j];
            }
        }

        this.b2 -= this.learningRate * db2;

        return loss;
    }
}
const nn = new NeuralNetwork(3, 8);



const dataset = [
    { x: [0, 0, 0], y: 0 },
    { x: [0, 0, 1], y: 1 },
    { x: [0, 1, 0], y: 1 },
    { x: [0, 1, 1], y: 1 },
    { x: [1, 0, 0], y: 1 },
    { x: [1, 0, 1], y: 1 },
    { x: [1, 1, 0], y: 1 },
    { x: [1, 1, 1], y: 1 }

];

for (let epoch = 0; epoch < 500000; epoch++) {

    let totalLoss = 0;

    dataset.forEach(data => {
        totalLoss += nn.train(data.x, data.y);
    });

    // Print every 500 steps
    if (epoch % 5000 === 0) {
        console.log("Epoch:", epoch, "Loss:", totalLoss);
    }
}

console.log("Final Predictions:");
dataset.forEach(data => {
    console.log(data.x, "=>", nn.forward(data.x));
});
