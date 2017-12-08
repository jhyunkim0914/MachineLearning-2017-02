
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


def one_hot(xy):
    dataY = []
    for i in range(0,len(xy)):
        temp = xy[i,-1]
        if(temp==0):
            _y = [1,0,0,0,0,0,0,0,0,0]
        elif(temp==1):
            _y = [0,1,0,0,0,0,0,0,0,0]
        elif(temp==2):
            _y = [0,0,1,0,0,0,0,0,0,0]
        elif(temp==3):
            _y = [0,0,0,1,0,0,0,0,0,0]
        elif(temp==4):
            _y = [0,0,0,0,1,0,0,0,0,0]
        elif(temp==5):
            _y = [0,0,0,0,0,1,0,0,0,0]
        elif(temp==6):
            _y = [0,0,0,0,0,0,1,0,0,0]
        elif(temp==7):
            _y = [0,0,0,0,0,0,0,1,0,0]
        elif(temp==8):
            _y = [0,0,0,0,0,0,0,0,1,0]
        elif(temp==9):
            _y = [0,0,0,0,0,0,0,0,0,1]
        dataY.append(_y)
    return dataY


# In[3]:


xy = np.loadtxt('test02.csv', delimiter=',', dtype=np.float32)

random.shuffle(xy)
dataX = xy[:, 0:-1]
dataY = xy[:, [-1]]
dataY_one_hot = one_hot(xy)


# In[5]:


feature = 6
nb_classes = 10
learning_rate = 0.001
training_epochs = 2000


# In[6]:


# train/test split
train_size = int(len(dataY_one_hot) * 0.8)
test_size = len(dataY_one_hot) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY_one_hot[0:train_size]), np.array(
    dataY_one_hot[train_size:len(dataY_one_hot)])


# In[7]:


print(trainY)


# In[8]:


X = tf.placeholder(tf.float32, [None, feature])
Y = tf.placeholder(tf.float32, [None, nb_classes])


# In[9]:


# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[feature, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[256, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L4, W5) + b5


# In[10]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[11]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[14]:


for epoch in range(training_epochs):

    feed_dict = {X: trainX, Y: trainY, keep_prob: 0.7}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
    
print('Learning Finished!')


# In[15]:


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: testX, Y: testY, keep_prob: 1}))

