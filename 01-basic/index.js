/**
 * Simple Neural Network for XOR
 * 
 * Architecture:
 * 2 inputs → 2 hidden neurons → 1 output
 */
class NeuralNetwork {

    /**
     * Create neural network and initialize weights randomly
     */
    constructor() {

        /**
         * Hidden layer weights
         * Matrix 2x2
         * 
         * W1[0] → weights for first hidden neuron
         * W1[1] → weights for second hidden neuron
         */
        this.W1 = [
            [Math.random(), Math.random()],
            [Math.random(), Math.random()]
        ];

        /**
         * Hidden layer bias (2 neurons → 2 biases)
         */
        this.b1 = [0, 0];

        /**
         * Output layer weights (2 hidden neurons → 1 output)
         */
        this.W2 = [Math.random(), Math.random()];

        /**
         * Output bias
         */
        this.b2 = 0;

        /**
         * Learning rate controls how fast we update weights
         */
        this.learningRate = 0.1;
    }

    /**
     * ReLU activation
     * If value < 0 → return 0
     * Else return value
     * 
     * @param {number} x
     * @returns {number}
     */
    relu(x) {
        return Math.max(0, x);
    }

    /**
     * Derivative of ReLU
     * Needed for backpropagation
     * 
     * @param {number} x
     * @returns {number}
     */
    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    /**
     * Sigmoid activation
     * Squashes number between 0 and 1
     * 
     * @param {number} x
     * @returns {number}
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivative of sigmoid
     * 
     * @param {number} x
     * @returns {number}
     */
    sigmoidDerivative(x) {
        const s = this.sigmoid(x);
        return s * (1 - s);
    }

    /**
     * Forward pass
     * 
     * This calculates prediction from input
     * 
     * @param {[number, number]} x - Input array [x1, x2]
     * @returns {number} - Prediction
     */
    forward(x) {

        // ----- Hidden Layer -----

        /**
         * z1 = weighted sum for hidden neurons
         */
        this.z1 = [
            this.W1[0][0] * x[0] + this.W1[0][1] * x[1] + this.b1[0],
            this.W1[1][0] * x[0] + this.W1[1][1] * x[1] + this.b1[1]
        ];

        /**
         * Apply ReLU activation
         */
        this.a1 = [
            this.relu(this.z1[0]),
            this.relu(this.z1[1])
        ];

        // ----- Output Layer -----

        /**
         * Weighted sum for output
         */
        this.z2 =
            this.W2[0] * this.a1[0] +
            this.W2[1] * this.a1[1] +
            this.b2;

        /**
         * Final output after sigmoid
         */
        this.a2 = this.sigmoid(this.z2);

        return this.a2;
    }

    /**
     * Train on single data point
     * 
     * @param {[number, number]} x - input
     * @param {number} y - correct answer
     */
    train(x, y) {

        // Step 1: Forward pass
        const output = this.forward(x);

        /**
         * Step 2: Calculate error
         * Using Mean Squared Error derivative
         */
        const lossGradient = 2 * (output - y);

        /**
         * Step 3: Output layer gradient
         */
        const dz2 = lossGradient * this.sigmoidDerivative(this.z2);

        /**
         * Gradients for output weights
         */
        const dW2 = [
            dz2 * this.a1[0],
            dz2 * this.a1[1]
        ];

        const db2 = dz2;

        /**
         * Step 4: Backprop to hidden layer
         */
        const dz1 = [
            dz2 * this.W2[0] * this.reluDerivative(this.z1[0]),
            dz2 * this.W2[1] * this.reluDerivative(this.z1[1])
        ];

        const dW1 = [
            [dz1[0] * x[0], dz1[0] * x[1]],
            [dz1[1] * x[0], dz1[1] * x[1]]
        ];

        const db1 = dz1;

        /**
         * Step 5: Update weights
         */
        for (let i = 0; i < 2; i++) {

            this.W2[i] -= this.learningRate * dW2[i];
            this.b1[i] -= this.learningRate * db1[i];

            for (let j = 0; j < 2; j++) {
                this.W1[i][j] -= this.learningRate * dW1[i][j];
            }
        }

        this.b2 -= this.learningRate * db2;
    }
}

const nn = new NeuralNetwork();

/**
 * XOR dataset
 */
const dataset = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },

];

/**
 * Train for many iterations
 */
for (let epoch = 0; epoch < 50000; epoch++) {
    dataset.forEach(data => {
        nn.train(data.x, data.y);
    });
}

/**
 * Test results
 */
dataset.forEach(data => {
    console.log(data.x, "=>", nn.forward(data.x));
});
