import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
import tensorflow as tf
import numpy as np
import tf_util
import ops
from pointnet_util import pointnet_sa_module, pointnet_fp_module

## Hyper parameters for voxel
#voxel_grid = 32
voxel_channel = 1
voxel_network_depth = 4
voxel_start_channel_num = 32
voxel_channel_axis = 4
voxel_conv_size = (3, 3, 3)
voxel_pool_size = (2, 2, 2)
voxel_action = 'concat'
voxel_class_num = 21+1

def placeholder_inputs(batch_size, v_size):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, v_size, v_size, v_size, 1))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, v_size, v_size, v_size))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, v_size, v_size, v_size))
    return pointclouds_pl, labels_pl, smpws_pl

def build_down_block(inputs, name, down_outputs, first=False,TPS=False, is_training=True, bn_decay=None):
    out_num = voxel_start_channel_num if first else 2 * \
              inputs.shape[voxel_channel_axis].value
    
    conv1 = tf_util.conv3d(inputs, out_num, voxel_conv_size, name+'/conv1', bn=True, is_training=is_training, bn_decay=bn_decay)
    #if TPS == True:
    #    conv1= self.transform.Encoder(conv1,conv1)
    conv2 = tf_util.conv3d(conv1, out_num, voxel_conv_size, name+'/conv2', bn=True, is_training=is_training, bn_decay=bn_decay)
    down_outputs.append(conv2)
    pool = ops.pool(conv2, voxel_pool_size, name +
                    '/pool')
    return pool

def build_bottom_block( inputs, name, is_training=True, bn_decay=None):
    out_num = inputs.shape[voxel_channel_axis].value
    conv1 = tf_util.conv3d(inputs, 2*out_num, voxel_conv_size, name+'/conv1', bn=True, is_training=is_training, bn_decay=bn_decay)
    conv2 = tf_util.conv3d(conv1, out_num, voxel_conv_size, name+'/conv2', bn=True, is_training=is_training, bn_decay=bn_decay)
    return conv2

def deconv_func():
    return getattr(ops, "deconv")

def conv_func():
    return getattr(ops, "conv")

def build_up_block(inputs, down_inputs, name, final=False,Decoder=False,is_training=True,bn_decay=None):
    out_num = inputs.shape[voxel_channel_axis].value
    conv1 = deconv_func()(
        inputs, out_num, voxel_conv_size, name+'/conv1',
        action=voxel_action, is_training=is_training, bn_decay=bn_decay)
    conv1 = tf.concat(
        [conv1, down_inputs], voxel_channel_axis, name=name+'/concat')
    conv2 = tf_util.conv3d(conv1, out_num, voxel_conv_size, name+'/conv2', bn=True, is_training=is_training, bn_decay=bn_decay)
    #if Decoder == True:
    #    conv2 = self.transform.Decoder(conv2,conv2)
    out_num = voxel_class_num if final else out_num/2
    conv3 = tf_util.conv3d(conv2, out_num, voxel_conv_size, name+'/conv3', bn=(not final), is_training=is_training, bn_decay=bn_decay)
    return conv3

def get_model(inputs, num_class, is_training=True, bn_decay=None):
    outputs = inputs
    down_outputs = []
    for layer_index in range(voxel_network_depth-1):
        is_first = True if not layer_index else False
        name = 'down%s' % layer_index
        outputs = build_down_block(outputs, name, down_outputs, first=is_first,TPS = False, is_training=is_training, bn_decay=bn_decay)  
        print("down ",layer_index," shape ", outputs.get_shape())          
    outputs = build_bottom_block(outputs, 'bottom', is_training=is_training, bn_decay=bn_decay)
    for layer_index in range(voxel_network_depth-2, -1, -1):
        is_final = True if layer_index == 0 else False
        name = 'up%s' % layer_index
        down_inputs = down_outputs[layer_index]
        outputs = build_up_block(outputs, down_inputs, name,final=is_final,Decoder=False, is_training=is_training, bn_decay=bn_decay )
        print("up ",layer_index," shape ",outputs.get_shape())
    return outputs

def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    bsize = pred.get_shape()[0]
    classify_loss1 = tf.losses.sparse_softmax_cross_entropy(labels=label[0:bsize//2,...], logits=pred[0:bsize//2,...], weights=smpw[0:bsize//2,...])
    classify_loss2 = tf.losses.sparse_softmax_cross_entropy(labels=label[bsize//2:bsize,...], logits=pred[bsize//2:bsize,...], weights=smpw[bsize//2:bsize,...])
    classify_loss = classify_loss1 + 0.75*classify_loss2
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss, classify_loss1, classify_loss2

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
