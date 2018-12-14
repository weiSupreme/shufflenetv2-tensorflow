
# coding: utf-8

# In[ ]:


import tensorflow as tf
import json
import cv2
import numpy as np
from PIL import Image

from architecture import shufflenet


# # Load label decoding

# In[ ]:


with open('data/integer_encoding.json', 'r') as f:
    encoding = json.load(f)
    
with open('data/wordnet_decoder.json', 'r') as f:
    wordnet_decoder = json.load(f)


# In[ ]:


decoder = {i: wordnet_decoder[n] for n, i in encoding.items()}


# # Load an image

# In[ ]:


image = cv2.imread('panda.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)

Image.fromarray(image)


# # Load a trained model

# In[ ]:


tf.reset_default_graph()

raw_images = tf.placeholder(tf.uint8, [None, 224, 224, 3])
images = tf.to_float(raw_images)/255.0

logits = shufflenet(images, is_training=False, depth_multiplier='0.5')
probabilities = tf.nn.softmax(logits, axis=1)

ema = tf.train.ExponentialMovingAverage(decay=0.995)
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)


# # Predict

# In[ ]:


with tf.Session() as sess:
    saver.restore(sess, 'run00/model.ckpt-1331064')
    feed_dict = {raw_images: np.expand_dims(image, axis=0)}
    result = sess.run(probabilities, feed_dict)[0]


# In[ ]:


print('The most probable labels is:')
print(decoder[np.argmax(result)])

