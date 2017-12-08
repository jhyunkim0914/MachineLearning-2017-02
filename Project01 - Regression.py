
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random

pp = pprint.PrettyPrinter(width=20,indent=6)


# In[2]:


xy = np.loadtxt('test01.csv', delimiter=',', dtype=np.float32)

random.shuffle(xy)
dataX = xy[:, 0:-1]
dataY = xy[:, [-1]]


# In[3]:


# Make sure the shape and data are OK
print(dataX.shape, dataX, len(dataX))
print(dataY.shape, dataY, len(dataY))


# In[4]:


# train/test split
train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])


# In[5]:


print(trainX)


# In[6]:


feature = 6
learning_rate=0.005
training_epochs = 3000


# In[7]:


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None,feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([feature, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# In[8]:


# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

#weights & bias for nn layers (fully-connected)
nb_Ly1 = 256
nb_Ly2 = 256
nb_Ly3 = 256
nb_Ly4 = 256


W1 = tf.get_variable("W1", shape=[feature, nb_Ly1],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([nb_Ly1]))
L1 = tf.nn.relu(tf.matmul(X,W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[nb_Ly1, nb_Ly2],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([nb_Ly2]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[nb_Ly2, nb_Ly3],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_Ly3]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[nb_Ly3, nb_Ly4],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_Ly4]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[nb_Ly4, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L4, W5) + b5


# In[9]:


cost = tf.reduce_mean(tf.square(hypothesis-trainY))
optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[10]:


# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())


# In[11]:


for epoch in range(training_epochs):
    
    feed_dict = {X: trainX, Y: trainY, keep_prob: 0.7}
    c, _, resultY = sess.run([cost, optimizer, hypothesis], feed_dict=feed_dict)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
    
print('Learning Finished!')


# In[12]:


predY = sess.run([hypothesis], feed_dict={X: testX, Y: testY, keep_prob:1})


# In[13]:


plt.figure(figsize=(10,5))
plt.plot(testY)
plt.plot(predY[0])
plt.show()


# In[14]:


print(predY[0])


# In[15]:


plt.figure(figsize=(10,5))
plt.plot(trainY)
plt.plot(resultY)
plt.show()

