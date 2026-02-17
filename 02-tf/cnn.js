const tf = require('@tensorflow/tfjs-node');

/**
 * Create CNN model
 */
function createModel() {

  const model = tf.sequential();

  // Convolution layer
  model.add(tf.layers.conv2d({
    inputShape: [40, 100, 1], // height, width, channels
    filters: 16,
    kernelSize: 3,
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu'
  }));

  // Output layer (36 classes: A-Z + 0-9)
  model.add(tf.layers.dense({
    units: 36,
    activation: 'softmax'
  }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

console.log(createModel().summary()); // Test with dummy input