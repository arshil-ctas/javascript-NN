// Input (x1, x2)
//      ↓
// Output = Sigmoid(w1*x1 + w2*x2 + b)


/* 
y=sigmoid(w1x1+w2x2+b)
*/
/**
 * Single layer neural network (no hidden layer)
 * 
 * This CANNOT solve XOR.
 */
class SingleLayerNN {

    constructor() {
        this.w1 = Math.random();
        this.w2 = Math.random();
        this.b = 0;

        this.learningRate = 0.1;
    }

    /**
     * Sigmoid activation
     * @param {number} x
     * @returns {number}
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivative of sigmoid
     * @param {number} x
     * @returns {number}
     */
    sigmoidDerivative(x) {
        const s = this.sigmoid(x);
        return s * (1 - s);
    }

    /**
     * Forward pass
     * @param {[number, number]} x
     * @returns {number}
     */
    forward(x) {
        this.z = this.w1 * x[0] + this.w2 * x[1] + this.b;
        this.a = this.sigmoid(this.z);
        return this.a;
    }

    /**
     * Train on one sample
     * @param {[number, number]} x
     * @param {number} y
     */
    train(x, y) {

        const output = this.forward(x);

        // Loss derivative
        const lossGradient = 2 * (output - y);

        const dz = lossGradient * this.sigmoidDerivative(this.z);

        // Gradients
        const dw1 = dz * x[0];
        const dw2 = dz * x[1];
        const db = dz;

        // Update
        this.w1 -= this.learningRate * dw1;
        this.w2 -= this.learningRate * dw2;
        this.b -= this.learningRate * db;
    }
}


const nn = new SingleLayerNN();

const dataset = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 }
];

for (let epoch = 0; epoch < 5000; epoch++) {
    dataset.forEach(data => {
        nn.train(data.x, data.y);
    });
}

dataset.forEach(data => {
    console.log(data.x, "=>", nn.forward(data.x));
});


/* 
NOTE :
Linear models can’t solve XOR

Neural networks need hidden layers


*/