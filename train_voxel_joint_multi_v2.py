import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # provider
import provider
import tf_util
import pc_util
import scannet_dataset
import suncg_dataset_multi
import math

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg_voxel', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_vol24_joint_multiv_sample_newa/', help='Log dir [default: log]')
### Start with isolated trained model
parser.add_argument('--restore_dir', default='models/VOLII/best_model.ckpt', help='Restore dir [default: log]')
parser.add_argument('--num_point', type=int, default=12288, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=201*3, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
restore_dir = FLAGS.restore_dir

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = FLAGS.model+'.py'
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 21
V_SIZE = 24

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR,'data','scannet_data_pointnet2')
TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train')
TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')
SUNCG_DATASET = suncg_dataset_multi.SuncgDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', rep='voxel2')

def pc_normalize_batch(pc):
    bsize = pc.shape[0]
    newpc = []
    for i in range(bsize):
        curpc = pc[i]
        centroid = np.mean(curpc, axis=0)
        curpc = curpc - centroid
        m = np.max(np.sqrt(np.sum(curpc**2, axis=1)))
        curpc = curpc / m
        newpc.append(curpc)
    return np.array(newpc)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, V_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Set learning rate and optimizer
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # -------------------------------------------
            # Get model and loss on multiple GPU devices
            # -------------------------------------------
            # Allocating variables on CPU first will greatly accelerate multi-gpu training.
            # Ref: https://github.com/kuza55/keras-extras/issues/21
            print "--- Get model and loss"
            # Get model and loss 
            pred = MODEL.get_model(pointclouds_pl, NUM_CLASSES, is_training_pl, bn_decay=bn_decay)
            
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d'%(0)), tf.name_scope('gpu_%d'%(0)) as scope:
                    pred = MODEL.get_model(pointclouds_pl, NUM_CLASSES, is_training_pl, bn_decay=bn_decay)
                    _, loss1, loss2 = MODEL.get_loss(pred, labels_pl, smpws_pl)
                    losses = tf.get_collection('losses', scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    
                    grads = optimizer.compute_gradients(total_loss)
                    
            # Get training operator 
            train_op = optimizer.apply_gradients(grads, global_step=batch)

            correct = tf.equal(tf.argmax(pred, 4), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)
            
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        if restore_dir != 'None':
            saver.restore(sess, restore_dir)
        else:
            print ("issue here! Must have a pretrained model")
            sys.exit(0)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
	       'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'loss1': loss1,
               'loss2': loss2,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        ### Evaluate first
        best_acc = eval_whole_scene_one_epoch(sess, ops, test_writer)
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
	    if (epoch+1)%5==0:
		acc = eval_whole_scene_one_epoch(sess, ops, test_writer)
                if acc > best_acc:
                    best_acc = acc
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
	batch_smpw[i,:] = smpw
        
	dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
	batch_data[i,drop_idx,:] = batch_data[i,0,:]
	batch_label[i,drop_idx] = batch_label[i,0]
	batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/(BATCH_SIZE // 2)
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    loss_sum1 = 0
    loss_sum2 = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * (BATCH_SIZE // 2)
        end_idx = (batch_idx+1) * (BATCH_SIZE // 2)

        ### Get data from pc process
        batch_data, batch_smpw = SUNCG_DATASET.wait_other()
        ###Convert it to voxel
        batch_data_norm = pc_normalize_batch(batch_data)
        batch_data_temp = pc_util.point_cloud_label_to_volume_batch_exact(batch_data_norm, vsize=V_SIZE, flatten=True)

        batch_data_vol = np.zeros((BATCH_SIZE, V_SIZE, V_SIZE, V_SIZE, 1))
        batch_data_vol[0:BATCH_SIZE//2,:,:,:,:] = batch_data_temp
        feed_dict = {ops['pointclouds_pl']: batch_data_vol,
                     ops['is_training_pl']: False}
        pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
        pred_val = np.expand_dims(np.argmax(pred_val, 4), -1)
        pred_val = pred_val[0:BATCH_SIZE//2,:,:,:,:]
        ###Convert it back to pc
        pred_val = np.clip(pc_util.volume_topc_batch_exact(pred_val, batch_data_norm, vsize=V_SIZE) - 1, a_min=0, a_max=None) ### Clip the label in case of Nan in training
        batch_data_extra, batch_label_extra, batch_smpw_extra = SUNCG_DATASET.ready(batch_data, np.squeeze(pred_val), batch_smpw, TRAIN_DATASET.labelweights)
        
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        batch_data = np.concatenate([batch_data, batch_data_extra], 0)
        batch_label = np.concatenate([batch_label, batch_label_extra], 0)
        batch_smpw = np.concatenate([batch_smpw, batch_smpw_extra], 0)
        
        # Augment batched point clouds by rotation
	aug_data = provider.rotate_point_cloud_z(batch_data)
        ###Convert it to voxel
        aug_data_vol, batch_label_vol, batch_smpw_vol = pc_util.point_cloud_label_to_volume_batch(pc_normalize_batch(aug_data), batch_label+1, batch_smpw, vsize=V_SIZE, flatten=True)
        
        feed_dict = {ops['pointclouds_pl']: aug_data_vol,
                     ops['labels_pl']: batch_label_vol,
		     ops['smpws_pl']:batch_smpw_vol,
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, loss_val1, loss_val2, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['loss1'], ops['loss2'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        ### Change the voxel back to pc
        pred_val = np.argmax(pred_val, 4)
        pred_val, batch_label, batch_smpw, _, _ = pc_util.volume_topc_batch(pred_val, batch_label_vol, batch_smpw_vol)
        for i in range(len(pred_val)):
            pred_val[i] -= 1
        for i in range(len(batch_label)):
            batch_label[i] -= 1

        for i in range(len(pred_val)):
            correct = np.sum(pred_val[i] == batch_label[i])
            total_correct += correct
            total_seen += pred_val[i].shape[0]
        loss_sum += loss_val
        loss_sum1 += loss_val1
        loss_sum2 += loss_val2
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('mean loss1: %f' % (loss_sum1 / 10))
            log_string('mean loss2: %f' % (loss_sum2 / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

# evaluate on whole scenes to generate numbers provided in the paper
# For consistency, convert it back to pointcloud and evaluated with the code provided in pointnet2
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE_SCENE))
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----'%(EPOCH_CNT))

    labelweights = np.zeros(21)
    labelweights_vox = np.zeros(21)
    is_continue_batch = False
    
    extra_batch_data = np.zeros((0,NUM_POINT,3))
    extra_batch_label = np.zeros((0,NUM_POINT))
    extra_batch_smpw = np.zeros((0,NUM_POINT))
    for batch_idx in range(num_batches):
	if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
	    batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
	    batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
	    batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
	else:
	    batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
	    batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
	    batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
	    batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)
	if batch_data.shape[0]<BATCH_SIZE:
	    is_continue_batch = True
	    continue
	elif batch_data.shape[0]==BATCH_SIZE:
	    is_continue_batch = False
	    extra_batch_data = np.zeros((0,NUM_POINT,3))
    	    extra_batch_label = np.zeros((0,NUM_POINT))
    	    extra_batch_smpw = np.zeros((0,NUM_POINT))
	else:
	    is_continue_batch = False
	    extra_batch_data = batch_data[BATCH_SIZE:,:,:]
    	    extra_batch_label = batch_label[BATCH_SIZE:,:]
    	    extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
	    batch_data = batch_data[:BATCH_SIZE,:,:]
    	    batch_label = batch_label[:BATCH_SIZE,:]
    	    batch_smpw = batch_smpw[:BATCH_SIZE,:]

        aug_data = batch_data
        aug_data_vol, batch_label_vol, batch_smpw_vol = pc_util.point_cloud_label_to_volume_batch(pc_normalize_batch(aug_data), batch_label+1, batch_smpw, vsize=V_SIZE, flatten=True)
        feed_dict = {ops['pointclouds_pl']: aug_data_vol,
                     ops['labels_pl']: batch_label_vol,
	  	     ops['smpws_pl']: batch_smpw_vol,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
	test_writer.add_summary(summary, step)
        ### Change the voxel back to pc
        pred_val = np.argmax(pred_val, 4)
        pred_val, batch_label, batch_smpw, _, aug_data = pc_util.volume_topc_batch(pred_val, batch_label_vol, batch_smpw_vol)
        for i in range(len(pred_val)):
            pred_val[i] -= 1
        for i in range(len(batch_label)):
            batch_label[i] -= 1
        for i in range(len(batch_label)):
            correct = np.sum((pred_val[i] == batch_label[i]) & (batch_label[i]>0) & (batch_smpw[i]>0)) # evaluate only on 20 categories but not unknown
            total_correct += correct
            total_seen += np.sum((batch_label[i]>0) & (batch_smpw[i]>0))
            loss_sum += loss_val
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label[i]==l) & (batch_smpw[i]>0))
                total_correct_class[l] += np.sum((pred_val[i]==l) & (batch_label[i]==l) & (batch_smpw[i]>0))
	for b in range(len(batch_label)):
            if (aug_data[b][batch_smpw[b]>0,:].shape)[0] == 0:
                continue
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b][batch_smpw[b]>0,:], np.concatenate((np.expand_dims(batch_label[b][batch_smpw[b]>0],1),np.expand_dims(pred_val[b][batch_smpw[b]>0],1)),axis=1), res=0.02)
	    total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
	    tmp,_ = np.histogram(uvlabel[:,0],range(22))
	    labelweights_vox += tmp
	    for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval whole scene point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
    log_string('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
    log_string('eval whole scene point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    caliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)
    log_string('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------'
    for l in range(1,NUM_CLASSES):
	per_class_str += 'class %d weight: %f, acc: %f; ' % (l,labelweights_vox[l-1],total_correct_class_vox[l]/float(total_seen_class_vox[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return caliacc

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
