# import dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# MNIST dataset parameters
NUM_CLASSES = 10 # total classes (0-9 digits)
NUM_FEATURES = 784 # data features (img shape: 28*28=784)

# Training parameters
LR = 0.001
TRAINING_STEPS = 3000
BATCH_SIZE = 250
DISPLAY_STEP = 100

# Networkk parametrs
n_hidden = 512

optimizer = tf.keras.optimizers.SGD(LR)

# Store layer's weights and bias

# a random value generator to initialize weights initially
random_normal = tf.initializers.RandomNormal()

weights =  {
    'h': tf.Variable(random_normal([NUM_FEATURES, n_hidden])),
    'out': tf.Variable(random_normal([n_hidden, NUM_CLASSES]))
}

biases = {
    'b': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([NUM_CLASSES]))
}


def neural_net(inputData):
    # Hidden fully connected layer with 512 neurons
    hidden_layer = tf.add(tf.matmul(inputData, weights['h']), biases['b'])
    # Apply sigmoid to hidden_layer output for non-linearity.
    hidden_layer = tf.nn.sigmoid(hidden_layer)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(hidden_layer, weights['out'] + biases['out'])
    # Apply softmax to normalize the logits to a probability distribution
    return tf.nn.softmax(out_layer)


def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector
    # our y_true[0] is equal to 1
    # in one hot vector format y_true[0] is equal to [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y_true = tf.one_hot(y_true, depth=NUM_CLASSES)
    # Clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


def run_optimization(x, y):
    # Wrap computation inside aa GradientTape for automatic differentiation
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)
    
    # Variables to update, trainable variables
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients
    gradients = g.gradient(loss, trainable_variables)

    # updatte weight adn biases following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# PREPARE DATASET
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# Flatten images to 1-D vector of 784 features (28*28)
x_train, x_test = x_train.reshape([-1, NUM_FEATURES]), x_test.reshape([-1, NUM_FEATURES])

# Normalize images value from [0, 255] to [0, 1]
x_train, x_test = x_train / 255. , x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(BATCH_SIZE).prefetch(1)

print(weights['h'].shape, weights['out'].shape)
print(biases['b'].shape, biases['out'].shape)

# run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(TRAINING_STEPS), 1):
    # run the optimization to update W and b values
    run_optimization(batch_x, batch_y)

    if step % DISPLAY_STEP == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print(f"Training epoch:{step}, Loss {loss}, Accuraacy: {acc}")
