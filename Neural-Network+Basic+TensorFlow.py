
# coding: utf-8

# # Neural-Network Basic TensorFlow

# Import Library

# In[1]:

import numpy as np
import tensorflow as tf


# Hyperparameters

# In[2]:

learning_rate    = 1e-4
input_dimension  = 4
neuron_per_layer = 20
output_dimension = input_dimension


# Create the placeholder for the input and output of the NN, (non trainable variable)

# In[3]:

INPUT  = tf.placeholder(tf.float32, [None,  input_dimension])
OUTPUT = tf.placeholder(tf.float32, [None, output_dimension])


# Create the Weight we are gonna modifiy to improve network accuracy (i'll set them to random value from a normal distribution with std_dev = 0.1 because it's a good stocastic inizialization)

# In[4]:

W_i = tf.Variable(tf.truncated_normal([input_dimension,neuron_per_layer], stddev=0.1), dtype=tf.float32)
b_i = tf.Variable(tf.truncated_normal([neuron_per_layer], stddev=0.1), dtype=tf.float32)

W   = tf.Variable(tf.truncated_normal([neuron_per_layer,neuron_per_layer], stddev=0.1), dtype=tf.float32)
B   = tf.Variable(tf.truncated_normal([neuron_per_layer], stddev=0.1), dtype=tf.float32)

W_o = tf.Variable(tf.truncated_normal([neuron_per_layer,output_dimension], stddev=0.1), dtype=tf.float32)
b_o = tf.Variable(tf.truncated_normal([output_dimension], stddev=0.1), dtype=tf.float32)


# Actual NN model

# In[5]:

a = tf.add(tf.matmul(INPUT,W_i),b_i)
a = tf.nn.sigmoid(a)
a = tf.add(tf.matmul(a,W),B)
a = tf.nn.sigmoid(a)
a = tf.add(tf.matmul(a,W_o),b_o)
output = tf.nn.sigmoid(a)


# Define the cost function we are gonna differentiate

# In[6]:

cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(OUTPUT,logits=output))


# Set the optimizer

# In[7]:

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Actually create the computation graph and initilize the variables

# In[8]:

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# ### Test the NN

# Create random data

# In[9]:

dummy_data = np.random.rand(1,input_dimension)

print(dummy_data)


# Calculate the output

# In[10]:

print(sess.run(output,feed_dict={INPUT:dummy_data}))


# ### Generate Data To Train The NN

# How many pair Input,Output are we gonna generate

# In[11]:

n_of_pair_of_data = 10000


# Generate NOT Pair of array

# In[12]:

X = np.random.randint(0,2,(n_of_pair_of_data,input_dimension))
Y = 1 - X

print("X\n%s"%X)
print("Y\n%s"%Y)


# ### Train the NN

# How many iteration of training we are gonna do

# In[13]:

n_of_iteration = 100
batch_size = 10


# Train loop and print output every 20 iteration

# In[14]:

for i in range(n_of_iteration):
    #train for batches
    for j in range(n_of_pair_of_data//batch_size):
        i1 = batch_size*j
        i2 = batch_size*(j+1)
        sess.run(optimizer,feed_dict={INPUT:X[i1:i2],OUTPUT:Y[i1:i2]})
    
    #print stuff
    if i % 10 == 0:
        print(i)
        j = np.random.randint(0,input_dimension - 1)
        res = sess.run(output,feed_dict={INPUT:X[j].reshape((1,input_dimension))})[0]
        print("result: \t%s"%res)
        #convert to 1 an 0 from probability
        res = np.array(res > 0.5,dtype=np.int)
        print("casted: \t%s"%res)
        print("original\t%s"%Y[j])


# ### Test the NN

# In[15]:

X_test = np.random.randint(0,2,(1,input_dimension))
res = sess.run(output,feed_dict={INPUT:X_test})
print(res)
res = np.array(res > 0.5,dtype=np.int)
print(res)
print(X_test)

