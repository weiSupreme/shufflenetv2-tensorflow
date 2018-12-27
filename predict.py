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
session_config.gpu_options.visible_device_list = '1'
session_config.gpu_options.allow_growth = True
with tf.Session(graph=tf.Graph(), config=session_config) as sess:
    tf.saved_model.loader.load(sess,['serve'],'./models/tzb/1545921290')
    graph = tf.get_default_graph()
    names=[n.name for n in tf.get_default_graph().as_graph_def().node]
    x=sess.graph.get_tensor_by_name('images:0')
    y=sess.graph.get_tensor_by_name('classes:0')
    imgl=os.listdir('train_images')
    for imgn in imgl:
        img=Image.open('train_images/'+imgn)
        bft=time.clock()
        img=img.convert('L')
        img=img.crop((145,153,481,489))
        image=np.array(img).reshape(336,336,1)
        c_ = sess.run(y, feed_dict={x: image})
        aft=time.clock()
        print(aft-bft, ' ', imgn, ' class: ', int2name[str(c_[0])])
        #break
