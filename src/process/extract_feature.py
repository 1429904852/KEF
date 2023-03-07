import tensorflow as tf
from tensorflow.contrib.slim import nets
import cv2
import numpy as np
import os

slim = tf.contrib.slim


base_path = "/KEF/"

X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='X')

with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
    net, endpoints = nets.resnet_v2.resnet_v2_152(inputs=X, is_training=False)

name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
feat = tf.get_default_graph().get_tensor_by_name('resnet_v2_152/block4/unit_3/bottleneck_v2/conv3/Conv2D:0')

pictures = os.listdir(base_path + "absa_data/twitter2015_images")

for picture in pictures:
    img = cv2.imread(base_path + 'absa_data/twitter2015_images/' + picture)
    try:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, [-1, img.shape[0], img.shape[1], 3])
        img = img.astype('float32')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint_path = base_path + 'pretrain_model/resnet_v2_152.ckpt'
            restorer = tf.train.Saver()
            restorer.restore(sess, checkpoint_path)
            feat_ = sess.run(feat, feed_dict={X: img})
            feat_ = np.reshape(feat_, [7 * 7, 2048])
            np.save(base_path + "absa_data/images2015_feature/" + picture.split(".")[0], feat_)
    except:
        print(picture)