"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    tf.summary.FileWriterCache.clear()

    # Input Layer
    # The methods in the layers module for creating convolutional and pooling layers
    # for two-dimensional image data expect input tensors to have a shape of [batch_size,
    # image_width, image_height, channels], defined as follows:
    #
    # batch_size - Size of the subset of examples to use when performing gradient descent during training.
    # image_width - Width of the example images.
    # image_height - Height of the example images.
    # channels - Number of color channels in the example images. For color images,
    # the number of channels is 3 (red, green, blue). For monochrome images, there is just 1 channel (black).
    #
    # Here, our MNIST dataset is composed of monochrome 28x28 pixel images,
    # so the desired shape for our input layer is [batch_size, 28, 28, 1].
    #
    # Reshape X to 4D tensor: [batch_size, width, height, channels]
    # MNIST images are 28*28 pixels, and have one color channel
    with tf.name_scope('reshape'):
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # We've indicated -1 for batch size, which specifies that this dimension should be dynamically computed
    # based on the number of input values in features["x"], holding the size of all other dimensions constant.
    # This allows us to treat batch_size as a hyperparameter that we can tune.
    # For example, if we feed examples into our model in batches of 5, features["x"] will contain 3,920 values
    # (one value for each pixel in each image), and input_layer will have a shape of [5, 28, 28, 1]

    # Convolutional Layer 1
    # We're connecting our first convolutional layer to input_layer, which has the shape [batch_size, 28, 28, 1].
    # Computes 32 features using a 5*5 filter with ReLU activation
    # kernel_size specifies the dimensions of the filters as [width, height] (here, [5, 5]).
    # The padding argument specifies one of two enumerated values (case-insensitive): valid (default value) or same.
    # To specify that the output tensor should have the same width
    # and height values as the input tensor, we set padding=same here,
    # which instructs TensorFlow to add 0 values to the edges of the
    # input tensor to preserve width and height of 28.
    # Without padding, a 5x5 convolution over a 28x28 tensor will produce a 24x24 tensor.
    # The activation argument specifies the activation function to apply to the output of the convolution.
    # Here, we specify ReLU activation with tf.nn.relu.
    # Input tensor shape: [batch_size, 28, 28, 1]
    # Output tensor shape: [batch_size, 28, 28, 32]
    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer 1
    # First max pooling layer with a 2*2 filter and a stride of 2
    # We set a stride of 2, which indicates that the subregions extracted
    # by the filter should be separated by 2 pixels
    # in both the width and height dimensions
    # Input tensor shape : [batch_size, 28, 28, 32]
    # Output tensor shape : [batch_size, 14, 14, 32]
    with tf.name_scope('pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer 2
    # Computes 64 features using a 5*5 filter with ReLU activation
    # Padding is added to preserve width and height
    # Input tensor shape: [batch_size, 14, 14, 32]
    # Output tensor shape: [batch_size, 14, 14, 64]
    with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer 2
    # Second max pooling layer with a 2*2 filter and a stride of 2
    # Input tensor shape : [batch_size, 14, 14, 64]
    # Output tensor shape : [batch_size, 7, 7, 64]
    with tf.name_scope('pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # In the reshape() operation above, the -1 signifies that the
    # batch_size dimension will be dynamically calculated
    # based on the number of examples in our input data.
    # Input tensor shape : [batch_size, 7, 7, 64]
    # Output tensor shape : [batch_size, 7*7*64]
    with tf.name_scope('pool2_flat'):
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input tensor shape : [batch_size, 7*7*64]
    # Output tensor shape : [batch_size, 1024]
    with tf.name_scope('dense'):
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # To help improve the results of our model, we also apply dropout regularization to our dense layer;
    # 0.6 probability that element will be kept and 0.4 that it will be discarded in training
    # The training argument takes a boolean specifying whether or not the model
    # is currently being run in training mode;
    # dropout will only be performed if training is True. Here,
    # we check if the mode passed to our model function cnn_model_fn is TRAIN mode.
    # dropout = tf.layers.dropout(
    #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # If the activation function is not specified, it is linear activation function
    # The final layer in our neural network is the logits layer, which will
    # return the raw values for our predictions.
    # Input tensor shape: [batch_size, 1024]
    # Output shape: [batch_size, 10]
    with tf.name_scope('logits'):
        logits = tf.layers.dense(inputs=dense, units=10)

    # Let's convert these raw values into two different formats that our model function can return:
    # The predicted class for each example: a digit from 0–9.
    # The probabilities for each possible target class for each example: the probability
    # that the example is a 0, is a 1, is a 2, etc.
    # The input argument specifies the tensor from which to extract maximum values—here logits.
    # The axis argument specifies the axis of the input tensor along which to find the greatest value.
    # Here, we want to find the largest value along the dimension with index of 1,
    # which corresponds to our predictions.
    # We can derive probabilities from our logits layer by applying softmax activation using tf.nn.softmax.
    # We use the name argument to explicitly name this operation softmax_tensor, so we can reference it later.
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # We compile our predictions in a dict, and return an EstimatorSpec object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # For both training and evaluation, we need to define a loss function that measures
    # how closely the model's predictions match the target classes.
    # For multiclass classification problems like MNIST, cross entropy is typically used as the loss metric.
    # Instead of using onehot encoding and later using softmax_cross_entropy,
    # we are here directly using sparse_softmax_cross_entropy.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    sess = tf.InteractiveSession()

    tf.summary.scalar("loss", loss)

    writer = tf.summary.FileWriter('/tmp/mnist_convnet')
    writer.add_graph(sess.graph)
    # writer.add_summary(loss)

    # Configure the Training Op (for TRAIN mode)
    # Optimize this loss value during training. We'll use a learning rate of 0.001 and
    # stochastic gradient descent as the optimization algorithm.

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (For EVAL mode)
    # To add accuracy metric in our model
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # We store the training feature data (the raw pixel values for 55,000 images of hand-drawn digits)
    # and training labels (the corresponding value from 0–9 for each image)
    # as numpy arrays in train_data and train_label.
    # Similarly, we store the evaluation feature data (10,000 images) and
    # evaluation labels in eval_data and eval_labels, respectively.
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the estimator (a TensorFlow class for performing high-level model training,
    # evaluation, and inference) for our model.
    # The model_fn argument specifies the model function to use for training,
    # evaluation, and prediction; we pass it the cnn_model_fn we created.
    # The model_dir argument specifies the directory where model data (checkpoints) will be saved.
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the probability values from the softmax layer of our CNN i.e.
    # Log the values in the "Softmax" tensor with label "probabilities".
    # We store a dict of the tensors we want to log in tensors_to_log.
    # Each key is a label of our choice that will be printed in the log output,
    # and the corresponding label is the name of a Tensor in the TensorFlow graph.
    # Here, our probabilities can be found in softmax_tensor,
    # the name we gave our softmax operation earlier when we generated the probabilities in cnn_model_fn.
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # We set every_n_iter=50, which specifies that probabilities should be logged after every 50 steps of training.
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    # In the numpy_input_fn call, we pass the training feature data and labels to x (as a dict) and y, respectively.
    # We set a batch_size of 100 (which means that the model will train on minibatches of 100 examples at each step)
    # num_epochs=None means that the model will train until the specified number of steps is reached.
    # We also set shuffle=True to shuffle the training data.
    # In the train call, we set steps=20000 (which means the model will train for 20,000 steps total).
    # We pass our logging_hook to the hooks argument, so that it will be triggered during training.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # To create eval_input_fn, we set num_epochs=1, so that the model evaluates
    # the metrics over one epoch of data and returns the result.
    # We also set shuffle=False to iterate through the data sequentially.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
