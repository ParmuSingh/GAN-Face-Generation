import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

RESTORE_SAVED_MODEL = False
SAVE_MODEL = True

base_path = "YOU_PATH_HERE" # The folder where you data is stored.

data = []

# Mounting training data from base_path.
for i in range(503):
	path = base_path + str(i) + '.png'
	
	try:
		img = Image.open(path)
		img = np.asarray(img)
		img = np.resize(img, [112*112])
		data.append(img)
	except:
		print("err")
		continue

# This function is used to provide the generator some random input.
def noise(batch_size):
    return np.random.uniform(-1, 1, (batch_size, 100))

# Hyper Parameters.
learning_rate = 0.001
batch_size = 128

#Placehoders.
input = tf.placeholder('float', [None, 100])
real_data = tf.placeholder('float', [None, 112*112])

# Generator
def generator(x):
    weights = {
        'hl1' : tf.get_variable(name='w_hl1',shape=[100, 200],initializer=tf.contrib.layers.xavier_initializer()),
        'hl2' : tf.get_variable(name='w_hl2',shape=[200, 300],initializer=tf.contrib.layers.xavier_initializer()),
        'hl3' : tf.get_variable(name='w_hl3',shape=[300, 500],initializer=tf.contrib.layers.xavier_initializer()),
        'ol'  : tf.get_variable(name='w_ol', shape=[500, 112*112],initializer=tf.contrib.layers.xavier_initializer())
    }
    biases = {
        'hl1' : tf.get_variable(name='b_hl1',shape=[200],initializer=tf.contrib.layers.xavier_initializer()),
        'hl2' : tf.get_variable(name='b_hl2',shape=[300],initializer=tf.contrib.layers.xavier_initializer()),
        'hl3' : tf.get_variable(name='b_hl3',shape=[500],initializer=tf.contrib.layers.xavier_initializer()),
        'ol'  : tf.get_variable(name='b_ol',shape=[112*112],initializer=tf.contrib.layers.xavier_initializer())
    }
    
    hl1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['hl1']), biases['hl1']))
    hl2 = tf.nn.leaky_relu(tf.add(tf.matmul(hl1, weights['hl2']), biases['hl2']))
    hl3 = tf.nn.leaky_relu(tf.add(tf.matmul(hl2, weights['hl3']), biases['hl3']))
    ol = tf.add(tf.matmul(hl3, weights['ol']), biases['ol']) # No activation because sigmoid() is gets applied in sigmoid_cross_entropy_with_logits() later.

    return ol

# Discriminator
def discriminator(x):
    weights = {
        'hl1' : tf.get_variable(name='w_hl1',shape=[112*112, 200],initializer=tf.contrib.layers.xavier_initializer()),
        'hl2' : tf.get_variable(name='w_hl2',shape=[200, 400],initializer=tf.contrib.layers.xavier_initializer()),
        'hl3' : tf.get_variable(name='w_hl3',shape=[400, 200],initializer=tf.contrib.layers.xavier_initializer()),
        'ol'  : tf.get_variable(name='w_ol',shape=[200, 1],initializer=tf.contrib.layers.xavier_initializer())
    }
    biases = {
        'hl1' : tf.get_variable(name='b_hl1',shape=[200],initializer=tf.contrib.layers.xavier_initializer()),
        'hl2' : tf.get_variable(name='b_hl2',shape=[400],initializer=tf.contrib.layers.xavier_initializer()),
        'hl3' : tf.get_variable(name='b_hl3',shape=[200],initializer=tf.contrib.layers.xavier_initializer()),
        'ol'  : tf.get_variable(name='b_ol',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    }

    hl1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['hl1']), biases['hl1']))
    hl2 = tf.nn.leaky_relu(tf.add(tf.matmul(hl1, weights['hl2']), biases['hl2']))
    hl3 = tf.nn.leaky_relu(tf.add(tf.matmul(hl2, weights['hl3']), biases['hl3']))
    ol = tf.add(tf.matmul(hl3, weights['ol']), biases['ol']) # No activation because sigmoid() is gets applied in sigmoid_cross_entropy_with_logits() later.
    
    return ol

# Variable scopes are used so we can later differentiate which weigts to use. See G_train and D_train.
with tf.variable_scope("G"):
    G = generator(input)

with tf.variable_scope("D"):
    D_real = discriminator(real_data)


with tf.variable_scope("D", reuse = True):
    D_gen = discriminator(G)

generator_parameters = [x for x in tf.trainable_variables() if x.name.startswith('G/')]
discriminator_parameters = [x for x in tf.trainable_variables() if x.name.startswith('D/')]

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gen, labels=tf.ones_like(D_gen)))
D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gen, labels=tf.zeros_like(D_gen)))
D_total_loss = tf.add(D_fake_loss, D_real_loss)

G_train = tf.train.AdamOptimizer(learning_rate).minimize(G_loss,var_list=generator_parameters)
D_train = tf.train.AdamOptimizer(learning_rate).minimize(D_total_loss,var_list=discriminator_parameters)


sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

if RESTORE_SAVED_MODEL:
	saver = tf.train.import_meta_graph("YOU_PATH_HERE")
	saver.restore(sess, tf.train.latest_checkpoint('YOU_PATH_HERE'))



loss_g_function = []
loss_d_function = []

for epoch in range(100):
    ptr = 0
    for iteratiion in range(int(len(data)/batch_size)):
        real_batch = data[ptr : ptr + batch_size]
        # real_batch = 2*real_batch - [1.]*len(real_batch)
        _, d_err = sess.run([D_train, D_total_loss], feed_dict = {real_data : real_batch, input : noise(batch_size)})
        _, g_err = sess.run([G_train, G_loss], feed_dict = {input : noise(batch_size)})
        loss_g_function.append(g_err)
        loss_d_function.append(d_err)
    if epoch %50 == 0:
        print("Epoch = ", epoch)
        print("D_loss = ", d_err)
        print("G_loss = ", g_err)
        
        # test_noise = noise(1)
        # plt.imshow(np.reshape(sess.run(G, feed_dict = {input : test_noise})[0], [112, 112]), cmap='gray')
        # plt.show()


# Visualizing

test_noise = noise(1)

plt.subplot(2, 2, 1)
plt.plot(test_noise[0])
plt.title("Noise")
plt.subplot(2, 2, 2)
plt.imshow(np.reshape(sess.run(G, feed_dict = {input : test_noise})[0], [112, 112]), cmap='gray')
plt.title("Generated Image")
plt.subplot(2, 2, 3)
plt.plot(loss_d_function, 'r')
plt.xlabel("Epochs")
plt.ylabel("Discriminator Loss")
plt.title("D-Loss")
plt.subplot(2, 2, 4)
plt.plot(loss_g_function, 'b')
plt.xlabel("Epochs")
plt.ylabel("Generator Loss")
plt.title("G_Loss")
plt.show()

if SAVE_MODEL:
	save_path = saver.save(sess, "YOU_PATH_HERE")
	print("Model saved at : ", save_path)
