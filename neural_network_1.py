# This is a neural network that takes input as a vector of words present in any sentence and gives the output as to weather it is a 
# negative statement or a positive statement

# Step 0: importing the dependencies
import tensorflow as tf
from sentiment_calculator_1 import create_features_set_and_labels

# Step 1: Creating the data
# test_size is the ratio of data to be used for testing to total data
# currently set at 70%
data = create_features_set_and_labels('pos.txt', 'neg.txt', test_size = 0.7)
train_x = data[0]
train_y = data[1]
test_x = data[2]
test_y = data[3]

# Step 2: Declaring network variables
n_hidden_1 = 500
n_hidden_2 = 500
n_hidden_3 = 500

# Step 3: Create the placeholders
x = tf.placeholder(tf.float32, shape = [None, len(train_x[0])])

# layer 1
W1 = tf.Variable(tf.truncated_normal([len(train_x[0]), n_hidden_1]))
b1 = tf.Variable(tf.truncated_normal([n_hidden_1]))

# layer 2
W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]))
b2 = tf.Variable(tf.truncated_normal([n_hidden_2]))

# layer 3
W3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]))
b3 = tf.Variable(tf.truncated_normal([n_hidden_3]))

# layer 4
W4 = tf.Variable(tf.truncated_normal([n_hidden_3, len(train_y[0])]))
b4 = tf.Variable(tf.truncated_normal([len(train_y[0])]))

# target output
y_ = tf.placeholder(tf.float32, shape = [None, len(train_y[0])])

# Step 4: Creating the neural network

# layer 1
y1 = tf.add(tf.matmul(x, W1), b1)
y1 = tf.nn.relu(y1)

# layer 2
y2 = tf.add(tf.matmul(y1, W2), b2)
y2 = tf.nn.relu(y2)

# layer 3
y3 = tf.add(tf.matmul(y2, W3), b3)
y3 = tf.nn.relu(y3)

# final layer
y4 = tf.add(tf.matmul(y3, W4), b4)
y = tf.nn.relu(y4)

# Step 5: Declaring the hyper-parameters
total_epochs = 5

# Step 6: Running the neural network
cross_entropy = cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

c_old = 0.0
c_new = 0.0

for epoch in range(total_epochs):
	sess.run(optimizer, feed_dict = {x: train_x, y_: train_y})
	c_new = sess.run(cost, feed_dict = {x: train_x, y_: train_y})
	print("Epoch", epoch+1, "completed out of", total_epochs, "loss:", c_new)

	# Simple condition to stop useless computation
	if c_new == c_old:
		break
	c_old = c_new

# Step 7: Accuracy checking
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print("Accuracy:", accuracy.eval({x: test_x, y_: test_y}, session = sess))
