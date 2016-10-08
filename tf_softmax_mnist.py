# Softmax Regression using tensorflow.
import tensorflow as tf

# Download the mnist data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True)

# Input placeholder, 2-D tensor of floating-point nunbers.
# here None means that a dimension can be of any length.
x = tf.placeholder(tf.float32, [None, 784])

# Initialize both W and b as tensors full of zeros.
# Since we are going to learn W and b, it doesn't
# matter very much what they initial are.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Maichine Learning Model.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# New placeholder to input the correct answers.
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), \
	            reduction_indices=[1]))                                   
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training 1000 times, 100 for each loop.
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Testing accuracy using test images.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
