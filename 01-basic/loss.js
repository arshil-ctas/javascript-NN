/**
 * “How wrong is the network?”
 * Loss=(prediction−actual)^2
 * 
*/

/**
 * Neural Network with Loss Tracking
 */
class NeuralNetwork {

    constructor() {

        this.W1 = [
            [Math.random(), Math.random()],
            [Math.random(), Math.random()]
        ];

        this.b1 = [0, 0];

        this.W2 = [Math.random(), Math.random()];
        this.b2 = 0;

        this.learningRate = 0.1;
    }

    /**
     * ReLU activation
     * @param {number} x
     * @returns {number}
     */
    relu(x) {
        return Math.max(0, x);
    }

    /**
     * ReLU derivative
     * @param {number} x
     * @returns {number}
     */
    reluDerivative(x) {
        return x > 0 ? 1 : 0;
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
     * Sigmoid derivative
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

        this.z1 = [
            this.W1[0][0] * x[0] + this.W1[0][1] * x[1] + this.b1[0],
            this.W1[1][0] * x[0] + this.W1[1][1] * x[1] + this.b1[1]
        ];

        this.a1 = [
            this.relu(this.z1[0]),
            this.relu(this.z1[1])
        ];

        this.z2 =
            this.W2[0] * this.a1[0] +
            this.W2[1] * this.a1[1] +
            this.b2;

        this.a2 = this.sigmoid(this.z2);

        return this.a2;
    }

    /**
     * Train on one sample
     * @param {[number, number]} x
     * @param {number} y
     * @returns {number} loss
     */
    train(x, y) {

        const output = this.forward(x);

        // ----- LOSS -----
        const loss = Math.pow(output - y, 2);

        // ----- BACKPROP -----

        const lossGradient = 2 * (output - y);
        const dz2 = lossGradient * this.sigmoidDerivative(this.z2);

        const dW2 = [
            dz2 * this.a1[0],
            dz2 * this.a1[1]
        ];

        const db2 = dz2;

        const dz1 = [
            dz2 * this.W2[0] * this.reluDerivative(this.z1[0]),
            dz2 * this.W2[1] * this.reluDerivative(this.z1[1])
        ];

        const dW1 = [
            [dz1[0] * x[0], dz1[0] * x[1]],
            [dz1[1] * x[0], dz1[1] * x[1]]
        ];

        const db1 = dz1;

        // ----- UPDATE -----
        for (let i = 0; i < 2; i++) {

            this.W2[i] -= this.learningRate * dW2[i];
            this.b1[i] -= this.learningRate * db1[i];

            for (let j = 0; j < 2; j++) {
                this.W1[i][j] -= this.learningRate * dW1[i][j];
            }
        }

        this.b2 -= this.learningRate * db2;

        return loss;
    }
}


const nn = new NeuralNetwork();

const dataset = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 }
];

for (let epoch = 0; epoch < 50000; epoch++) {

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


/*
#5000 
Epoch: 0 Loss: 1.1428712843581357
Epoch: 2000 Loss: 0.005578339243404476
Epoch: 4000 Loss: 0.002007799833168429
Final Predictions:
[ 0, 0 ] => 0.030636066987877054
[ 0, 1 ] => 0.9856158721612605
[ 1, 0 ] => 0.985585707213227
[ 1, 1 ] => 0.011962328538149922

#50000
Epoch: 0 Loss: 1.1428712843581357
Epoch: 2000 Loss: 0.005578339243404476
Epoch: 4000 Loss: 0.002007799833168429
Final Predictions:
[ 0, 0 ] => 0.030636066987877054
[ 0, 1 ] => 0.9856158721612605
[ 1, 0 ] => 0.985585707213227
[ 1, 1 ] => 0.011962328538149922

#500_000
Epoch: 0 Loss: 1.0765855623685778
Epoch: 5000 Loss: 0.0014567971989660961
Epoch: 10000 Loss: 0.0006305447348068036
Epoch: 15000 Loss: 0.0003954099767634855
Epoch: 20000 Loss: 0.0002859492655113996
Epoch: 25000 Loss: 0.00022304816910602738
Epoch: 30000 Loss: 0.0001824156493943862
Epoch: 35000 Loss: 0.00015403100157383148
Epoch: 40000 Loss: 0.00013314342551501437
Epoch: 45000 Loss: 0.0001171279604043763
Epoch: 50000 Loss: 0.00010448984580386863
Epoch: 55000 Loss: 0.00009426006216422189
Epoch: 60000 Loss: 0.00008581623246406263
Epoch: 65000 Loss: 0.00007872541347187129
Epoch: 70000 Loss: 0.00007270459168185763
Epoch: 75000 Loss: 0.00006751359610967744
Epoch: 80000 Loss: 0.00006300187617492644
Epoch: 85000 Loss: 0.000059042592900863044
Epoch: 90000 Loss: 0.000055543778482355626
Epoch: 95000 Loss: 0.0000524283641775174
Epoch: 100000 Loss: 0.000049636640057712456
Epoch: 105000 Loss: 0.00004712036409865623
Epoch: 110000 Loss: 0.000044842843378897165
Epoch: 115000 Loss: 0.00004277175090824713
Epoch: 120000 Loss: 0.0000408782104952377
Epoch: 125000 Loss: 0.00003914314728096257
Epoch: 130000 Loss: 0.000037546696623823916
Epoch: 135000 Loss: 0.00003607228149495299
Epoch: 140000 Loss: 0.00003470813328434711
Epoch: 145000 Loss: 0.00003344094973793285
Epoch: 150000 Loss: 0.0000322606054316807
Epoch: 155000 Loss: 0.000031159098633560254
Epoch: 160000 Loss: 0.000030129637859972106
Epoch: 165000 Loss: 0.00002916410034764602
Epoch: 170000 Loss: 0.000028257718858808828
Epoch: 175000 Loss: 0.000027404885033303745
Epoch: 180000 Loss: 0.00002660130005215595
Epoch: 185000 Loss: 0.000025842001790327402
Epoch: 190000 Loss: 0.000025124566979800456
Epoch: 195000 Loss: 0.000024444984215328057
Epoch: 200000 Loss: 0.000023800548463276754
Epoch: 205000 Loss: 0.000023188399312966877
Epoch: 210000 Loss: 0.000022606721446862568
Epoch: 215000 Loss: 0.00002205266964896127
Epoch: 220000 Loss: 0.00002152513142504307
Epoch: 225000 Loss: 0.000021021126638085344
Epoch: 230000 Loss: 0.000020539862694030156
Epoch: 235000 Loss: 0.000020080192379139502
Epoch: 240000 Loss: 0.00001963981262815329
Epoch: 245000 Loss: 0.000019218095796352785
Epoch: 250000 Loss: 0.000018814047281554445
Epoch: 255000 Loss: 0.000018425949713711794
Epoch: 260000 Loss: 0.000018053317101311512
Epoch: 265000 Loss: 0.000017695367676531937
Epoch: 270000 Loss: 0.00001735098536910813
Epoch: 275000 Loss: 0.00001701959249207659
Epoch: 280000 Loss: 0.000016700023209392866
Epoch: 285000 Loss: 0.000016392574076106915
Epoch: 290000 Loss: 0.000016095679941083513
Epoch: 295000 Loss: 0.000015808997276612266
Epoch: 300000 Loss: 0.000015532494485480943
Epoch: 305000 Loss: 0.00001526522035109372
Epoch: 310000 Loss: 0.000015006876030680338
Epoch: 315000 Loss: 0.000014756936813957208
Epoch: 320000 Loss: 0.000014515056656193684
Epoch: 325000 Loss: 0.000014280854698892475
Epoch: 330000 Loss: 0.000014054075643431375
Epoch: 335000 Loss: 0.000013833830919267576
Epoch: 340000 Loss: 0.000013620623903414871
Epoch: 345000 Loss: 0.000013413674922116904
Epoch: 350000 Loss: 0.000013213022178278262
Epoch: 355000 Loss: 0.00001301773751079262
Epoch: 360000 Loss: 0.000012828406142624804
Epoch: 365000 Loss: 0.000012644255339688391
Epoch: 370000 Loss: 0.000012465274126321696
Epoch: 375000 Loss: 0.000012291205362513465
Epoch: 380000 Loss: 0.00001212178750227316
Epoch: 385000 Loss: 0.000011956898391848836
Epoch: 390000 Loss: 0.000011796400296167444
Epoch: 395000 Loss: 0.000011640149953681391
Epoch: 400000 Loss: 0.00001148773925597484
Epoch: 405000 Loss: 0.000011339350931041819
Epoch: 410000 Loss: 0.000011194645572424374
Epoch: 415000 Loss: 0.000011053512420141079
Epoch: 420000 Loss: 0.000010915806187209317
Epoch: 425000 Loss: 0.000010781432298541908
Epoch: 430000 Loss: 0.000010650337555961418
Epoch: 435000 Loss: 0.000010522144374304526
Epoch: 440000 Loss: 0.000010397122454575365
Epoch: 445000 Loss: 0.000010274993271592956
Epoch: 450000 Loss: 0.000010155561067224168
Epoch: 455000 Loss: 0.000010038896420267911
Epoch: 460000 Loss: 0.000009924782864851235
Epoch: 465000 Loss: 0.000009813235018544277
Epoch: 470000 Loss: 0.00000970413496199436
Epoch: 475000 Loss: 0.000009597367300505706
Epoch: 480000 Loss: 0.000009492893868397833
Epoch: 485000 Loss: 0.000009390634409742374
Epoch: 490000 Loss: 0.000009290444395311407
Epoch: 495000 Loss: 0.000009192384676907535
Final Predictions:
[ 0, 0 ] => 0.002536396573457591
[ 0, 1 ] => 0.9990076500337189
[ 1, 0 ] => 0.9990076416430291
[ 1, 1 ] => 0.000832762966609695

*/