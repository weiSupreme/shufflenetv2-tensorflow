import os
from PIL import Image
import tensorflow as tf
import numpy as np
import time

int2name={}
txt=open('/home/zw/Downloads/tiny-imagenet-200/name2int.txt')
lines=txt.readlines()
for line in lines:
    line_s=line.strip('\n').split(' ')
    int2name.update({line_s[0]:line_s[1]})
txt.close()

session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = '0'
session_config.gpu_options.allow_growth = True
with tf.Session(graph=tf.Graph(), config=session_config) as sess:
    tf.saved_model.loader.load(sess,['serve'],'./models/tzb/1545202445')
    graph = tf.get_default_graph()
    x=sess.graph.get_tensor_by_name('images:0')
    y=sess.graph.get_tensor_by_name('classes:0')
    imgl=os.listdir('models/tzb/test_images')
    for imgn in imgl:
        img=Image.open('models/tzb/test_images/'+imgn)
        bft=time.clock()
        img=img.convert('L')
        img=img.resize((336, 336),Image.ANTIALIAS)
        image=(np.array(img,dtype=float)/255.0).reshape(1,224,224,1)
        c_ = sess.run(y, feed_dict={x: image})
        aft=time.clock()
        print(aft-bft, ' ', imgn, ' class: ', int2name[str(c_[0])])
        #break
