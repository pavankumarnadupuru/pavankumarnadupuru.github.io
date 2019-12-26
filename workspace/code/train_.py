#!/usr/bin/env python
# coding: utf-8

# In[14]:


import struct
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras


# In[6]:


filename={}
filename['train_images']='/home/pavan/Desktop/_telugu/train-images-idx3-ubyte'
filename['train_labels']='/home/pavan/Desktop/_telugu/train-labels-idx1-ubyte'
filename['test_images']='/home/pavan/Desktop/_telugu/t10k-images-idx3-ubyte'
filename['test_labels']='/home/pavan/Desktop/_telugu/t10k-labels-idx1-ubyte'


# In[8]:


with open(filename['train_images'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    train_images=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

with open(filename['train_labels'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    train_labels=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

with open(filename['test_images'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    test_images=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

with open(filename['test_labels'], 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    test_labels=np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

print(f'The tensorflow version is : ',tf.__version__)


# In[11]:


print(train_images.shape)
print(train_labels.shape)


# In[24]:


class_names=[]
for i in range(16):
    class_names.append(str(i))


# In[12]:


print(test_images.shape)
print(test_labels.shape)


# In[18]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[50]:


train_images = train_images
test_images = test_images


# In[51]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[52]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 50)),
#     keras.layers.Dense(420, activation=tf.nn.relu),
#     keras.layers.Dense(210, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])


# In[53]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[55]:


model.fit(train_images, train_labels, epochs=1000)


# In[69]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[70]:


predictions=model.predict(test_images)


# In[72]:


predictions[0]


# In[80]:


for i in range(14):
    print(np.argmax(predictions[i]),end='\t')


# In[81]:


for i in range(14):
    print(test_labels[i],end='\t')


# In[ ]:




