import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # model
import provider
import tf_util
import pc_util
import scannet_dataset
import scannet_dataset_multi
#import suncg_dataset_multi
import suncg_dataset_multi
import math
from multiprocessing import Process, Queue

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_pc_joint/', help='Log dir [default: log]')
### Start from isolated model
parser.add_argument('--restore_dir1', default='models/PCIII/best_model.ckpt', help='Restore dir [default: log]')
parser.add_argument('--restore_dir2', default='models/PCII/best_model.ckpt', help='Restore dir [default: log]')
parser.add_argument('--restore_dir3', default='models/PCI/best_model.ckpt', help='Restore dir [default: log]')
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
NUM_POINT = [4096, 8192, 12288]
NUM_POINT_MORE = [4096, 4096, 4096, 8192, 8192, 12288]
NUM_REP = [3, 2, 1]
NUM_PATH = 3
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
restore_dir = [FLAGS.restore_dir1, FLAGS.restore_dir2, FLAGS.restore_dir3]

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

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR,'data','scannet_data_pointnet2')
TRAIN_DATASET = scannet_dataset_multi.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train')
TEST_DATASET_WHOLE_SCENE = []
for i in range(len(NUM_POINT)):
    TEST_DATASET_WHOLE_SCENE.append(scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT[i], split='test'))
SUNCG_DATASET = suncg_dataset_multi.SuncgDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', batch_size=BATCH_SIZE//2)

DATA_QUEUE = Queue(maxsize=100)
DATA_QUEUE_SUN = Queue(maxsize=100)

def data_producer_suncg():
    while True:
        if DATA_QUEUE_SUN.qsize() < 100:
            data = SUNCG_DATASET.get_next()
            DATA_QUEUE_SUN.put(data)

def data_producer():
    num_batches = len(TRAIN_DATASET)/(BATCH_SIZE // 2)
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    batch_idx = 0
    
    while True:
        if DATA_QUEUE.qsize() < 100:
            if batch_idx >= num_batches:
                batch_idx = 0
                train_idxs = np.arange(0, len(TRAIN_DATASET))
                np.random.shuffle(train_idxs)
            start_idx = batch_idx * (BATCH_SIZE // 2)
            end_idx = (batch_idx+1) * (BATCH_SIZE // 2)
            data = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
            DATA_QUEUE.put(data)
            batch_idx += 1

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
    ### Start the data processing queue here
    scannet_p = Process(target=data_producer)
    scannet_p.start()
    print ("started scannet data processing process")
    suncg_p = Process(target=data_producer_suncg)
    suncg_p.start()
    print ("started suncg data processing process")
    
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            pointclouds_pl_4096, labels_pl_4096, smpws_pl_4096 = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT[0], "4096")
            pointclouds_pl_8192, labels_pl_8192, smpws_pl_8192 = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT[1], "8192")
            pointclouds_pl_12288, labels_pl_12288, smpws_pl_12288 = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT[2], "12288")
            is_training_pl = tf.placeholder(tf.bool, shape=())
            data_select = tf.placeholder(tf.int32, shape=())
            pt = tf.cond(tf.less(data_select, 1), lambda: pointclouds_pl_4096, lambda: tf.cond(tf.less(data_select, 2), lambda: pointclouds_pl_8192, lambda: pointclouds_pl_12288))
            label = tf.cond(tf.less(data_select, 1), lambda: labels_pl_4096, lambda: tf.cond(tf.less(data_select, 2), lambda: labels_pl_8192, lambda: labels_pl_12288))
            smpw = tf.cond(tf.less(data_select, 1), lambda: smpws_pl_4096, lambda: tf.cond(tf.less(data_select, 2), lambda: smpws_pl_8192, lambda: smpws_pl_12288))
            
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
            pred, end_points = MODEL.get_model(pt, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d'%(0)), tf.name_scope('gpu_%d'%(0)) as scope:
                    pred, end_points = MODEL.get_model(pt, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
                    MODEL.get_loss(pred, label, smpw)
                    losses = tf.get_collection('losses', scope)
                    total_loss = tf.add_n(losses, name='total_loss')
        
                    # Get training operator 
                    grads = optimizer.compute_gradients(total_loss)
                    
            train_op = optimizer.apply_gradients(grads, global_step=batch)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=20)
        
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

        # Add the multi model part
        # claim just variables
        copy_to_model0_op = []
        copy_from_model0_op = []
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print ("Shared variables")
        for i in range(1, NUM_PATH+1): ## 3 different pcs
            copy_from_model0_op_i = []
            copy_to_model0_op_i = []
            for var in all_vars:
                #new_var_name = var.name.replace('model0', 'model%d' % i)
                new_var_name = ("pc%d/" % i) + var.name
                if var in tf.trainable_variables():
                    trainable = True
                else:
                    trainable = False
                new_var = tf.get_variable(new_var_name.split(':')[0], shape=var.shape, dtype=var.dtype, trainable=trainable)
                copy_from_model0_op_i.append(new_var.assign(var))
                copy_to_model0_op_i.append(var.assign(new_var))
            copy_to_model0_op.append(copy_to_model0_op_i)
            copy_from_model0_op.append(copy_from_model0_op_i)
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(len(NUM_POINT)):
            saver.restore(sess, restore_dir[i])
            sess.run(copy_from_model0_op[i])

        ops = {'pointclouds_pl1': pointclouds_pl_4096,
               'labels_pl1': labels_pl_4096,
	       'smpws_pl1': smpws_pl_4096,
               'pointclouds_pl2': pointclouds_pl_8192,
               'labels_pl2': labels_pl_8192,
	       'smpws_pl2': smpws_pl_8192,
               'pointclouds_pl3': pointclouds_pl_12288,
               'labels_pl3': labels_pl_12288,
	       'smpws_pl3': smpws_pl_12288,
               'data_select': data_select,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        ### Check for evaluation
        ### Previous best models
        best_acc = []
        for i in range(len(NUM_POINT)):
            print ("Testing pc with "+str(NUM_POINT[i]))
            sess.run(copy_to_model0_op[i])
            best_acc.append(eval_whole_scene_one_epoch(sess, ops, test_writer, TEST_DATASET_WHOLE_SCENE[i], NUM_POINT[i], i))
        
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer, copy_from_model0_op, copy_to_model0_op)

            ### Testing takes some time, start testing after 200 epoches
	    if epoch > 200:
                if (epoch+1)%5==0:
                    counter = 0
                    for i in range(len(NUM_POINT)):
                        print ("Testing pc with "+str(NUM_POINT[i]))
                        sess.run(copy_to_model0_op[i])
                        acc = eval_whole_scene_one_epoch(sess, ops, test_writer, TEST_DATASET_WHOLE_SCENE[i], NUM_POINT[i], i)
                        if acc > best_acc[i]:
                            best_acc[i] = acc
                            save_path = saver.save(sess, os.path.join(LOG_DIR+'model'+str(i+1), "best_model_for%01d_epoch_%03d.ckpt"%(i, epoch)))
                            log_string("Model saved in file: %s" % save_path)
            else:
                if (epoch+1)%20==0:
                    counter = 0
                    for i in range(len(NUM_POINT)):
                        print ("Testing pc with "+str(NUM_POINT[i]))
                        sess.run(copy_to_model0_op[i])
                        acc = eval_whole_scene_one_epoch(sess, ops, test_writer, TEST_DATASET_WHOLE_SCENE[i], NUM_POINT[i], i)
                        if acc > best_acc[i]:
                            best_acc[i] = acc
                            save_path = saver.save(sess, os.path.join(LOG_DIR+'model'+str(i+1), "best_model_for%01d_epoch_%03d.ckpt"%(i, epoch)))
                            log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                for i in range(len(NUM_POINT)):
                    sess.run(copy_to_model0_op[i])
                    save_path = saver.save(sess, os.path.join(LOG_DIR+'model'+str(i+1), "model_for%01d.ckpt"%i))
                    log_string("Model saved in file: %s" % save_path)

    ### Terminate the data preprocessing here
    scannet_p.terminate()
    suncg_p.terminate()
    
def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = []
    batch_label = []
    batch_smpw = []
    
    for i in range(len(NUM_POINT)):
        for j in range(NUM_REP[i]):
            batch_data.append(np.zeros((bsize, NUM_POINT[i], 3)))
            batch_label.append(np.zeros((bsize, NUM_POINT[i]), dtype=np.int32))
            batch_smpw.append(np.zeros((bsize, NUM_POINT[i]), dtype=np.float32))
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        counter = 0
        for j in range(len(NUM_POINT)):
            for k in range(NUM_REP[j]):
                batch_data[counter][i,...] = ps[counter]
                batch_label[counter][i,:] = seg[counter]
	        batch_smpw[counter][i,:] = smpw[counter]
        
	        dropout_ratio = np.random.random()*0.875 # 0-0.875
                drop_idx = np.where(np.random.random((ps[counter].shape[0]))<=dropout_ratio)[0]
	        batch_data[counter][i,drop_idx,:] = batch_data[counter][i,0,:]
	        batch_label[counter][i,drop_idx] = batch_label[counter][i,0]
	        batch_smpw[counter][i,drop_idx] *= 0
                counter += 1
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx, num_point):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, 3))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_smpw = np.zeros((bsize, num_point), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
	batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def add_empty(feed_dict, data_mode, ops):
    for i in range(len(NUM_POINT)):
        if i != data_mode:
            feed_dict[ops['pointclouds_pl'+str(i+1)]] = np.zeros([BATCH_SIZE, NUM_POINT[i], 3])
            feed_dict[ops['labels_pl'+str(i+1)]] = np.zeros([BATCH_SIZE, NUM_POINT[i]])
            feed_dict[ops['smpws_pl'+str(i+1)]] = np.zeros([BATCH_SIZE, NUM_POINT[i]])
    return feed_dict

def shuffle_batch(batch_data, idx):
    batch_data_temp = []
    batch = []
    for i in range(len(batch_data)):
        batch_data_temp.append(batch_data[i].copy())
    for i in range(len(idx)):
        batch.append(batch_data_temp[idx[i]][i,...])
    return np.stack(batch,0)

def shuffle_data(batch_data):
    ### Shuffle 0 and 1, 2 and 3
    chunk = len(batch_data[0][0])//3+1
    idx = np.array([0]*chunk + [1]*chunk + [2]*(len(batch_data[0][0]) - 2*chunk))
    np.random.shuffle(idx)

    for i in range(len(batch_data)):
        batch_data[i][0] = shuffle_batch([batch_data[i][0], batch_data[i][1], batch_data[i][2]], idx)

    #idx = np.arange(len(batch_data[0][0])*2)
    chunk = len(batch_data[0][0])//2
    idx = np.array([0]*chunk + [1]*chunk)
    np.random.shuffle(idx)

    for i in range(len(batch_data)):
        batch_data[i][3] = shuffle_batch([batch_data[i][3], batch_data[i][4]], idx)
        
    return batch_data

def train_one_epoch(sess, ops, train_writer, copy_from_model0_op, copy_to_model0_op):
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
    batch_idx_suncg = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * (BATCH_SIZE // 2)
        end_idx = (batch_idx+1) * (BATCH_SIZE // 2)
        while DATA_QUEUE_SUN.empty():
            pass
        temp_batch_data, batch_smpw = DATA_QUEUE_SUN.get()
        
        SUNCG_DATASET.check_gone(temp_batch_data[-1], batch_smpw[-1]) ## Only give 12288 points to voxel
        batch_data = []
        for i in range(len(temp_batch_data)):
            batch_data.append(np.zeros((BATCH_SIZE, NUM_POINT_MORE[i], 3)))
            batch_data[i][0:BATCH_SIZE//2,:,:] = temp_batch_data[i]

        pred_val = []
        counter = 0
        for i in range(len(NUM_POINT)):
            for j in range(NUM_REP[i]):
                sess.run(copy_to_model0_op[i])
                feed_dict = {}
                feed_dict[ops['is_training_pl']] = False
                feed_dict[ops['data_select']] = i
                feed_dict[ops['pointclouds_pl'+str(i+1)]] = batch_data[counter]
                temp_pred_val = sess.run(ops['pred'], feed_dict=add_empty(feed_dict, i, ops))
                pred_val.append(np.squeeze(np.argmax(temp_pred_val[0:BATCH_SIZE//2,...], 2)))
                counter += 1

        ### Combine with other sources here
        batch_data_extra, batch_label_extra, batch_smpw_extra = SUNCG_DATASET.ready(temp_batch_data, pred_val, batch_smpw, TRAIN_DATASET.labelweights)

        while DATA_QUEUE.empty():
            pass
        batch_data, batch_label, batch_smpw = DATA_QUEUE.get()
        shuffled_data = shuffle_data([batch_data, batch_label, batch_smpw])
        batch_data, batch_label, batch_smpw = shuffled_data[0], shuffled_data[1], shuffled_data[2]
        shuffled_data = shuffle_data([batch_data_extra, batch_label_extra, batch_smpw_extra])
        batch_data_extra, batch_label_extra, batch_smpw_extra = shuffled_data[0], shuffled_data[1], shuffled_data[2]
        ### Combine data
        counter = 0
        for i in range(len(NUM_POINT)):
            for j in range(NUM_REP[i]):
                if j == 0:
                    sess.run(copy_to_model0_op[i])
                    batch_data_temp = np.concatenate([batch_data[counter], batch_data_extra[counter]], 0)
                    batch_label_temp = np.concatenate([batch_label[counter], batch_label_extra[counter]], 0)
                    batch_smpw_temp = np.concatenate([batch_smpw[counter], batch_smpw_extra[counter]], 0)
        
	            aug_data = provider.rotate_point_cloud_z(batch_data_temp)
                    feed_dict = {ops['pointclouds_pl'+str(i+1)]: aug_data,
                                 ops['labels_pl'+str(i+1)]: batch_label_temp,
		                 ops['smpws_pl'+str(i+1)]:batch_smpw_temp,
                                 ops['data_select']:i,
                                 ops['is_training_pl']: is_training,}
                    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=add_empty(feed_dict, i, ops))
                    sess.run(copy_from_model0_op[i])
                    train_writer.add_summary(summary, step)
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum(pred_val == batch_label_temp)
                    total_correct += correct
                    total_seen += (BATCH_SIZE*NUM_POINT[i])
                    loss_sum += loss_val
                counter += 1

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10.0 / float(len(NUM_POINT))))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

# evaluate on whole scenes to generate numbers provided in the paper (same evaluation with pointnet2)
def eval_whole_scene_one_epoch(sess, ops, test_writer, test_dataset, num_point, mode):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(test_dataset))
    num_batches = len(test_dataset)

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
    
    extra_batch_data = np.zeros((0,num_point,3))
    extra_batch_label = np.zeros((0,num_point))
    extra_batch_smpw = np.zeros((0,num_point))
    for batch_idx in range(num_batches):
	if not is_continue_batch:
            batch_data, batch_label, batch_smpw = test_dataset[batch_idx]
	    batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
	    batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
	    batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
	else:
	    batch_data_tmp, batch_label_tmp, batch_smpw_tmp = test_dataset[batch_idx]
	    batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
	    batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
	    batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)
	if batch_data.shape[0]<BATCH_SIZE:
	    is_continue_batch = True
	    continue
	elif batch_data.shape[0]==BATCH_SIZE:
	    is_continue_batch = False
	    extra_batch_data = np.zeros((0,num_point,3))
    	    extra_batch_label = np.zeros((0,num_point))
    	    extra_batch_smpw = np.zeros((0,num_point))
	else:
	    is_continue_batch = False
	    extra_batch_data = batch_data[BATCH_SIZE:,:,:]
    	    extra_batch_label = batch_label[BATCH_SIZE:,:]
    	    extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
	    batch_data = batch_data[:BATCH_SIZE,:,:]
    	    batch_label = batch_label[:BATCH_SIZE,:]
    	    batch_smpw = batch_smpw[:BATCH_SIZE,:]

	aug_data = batch_data
        feed_dict = {ops['pointclouds_pl'+str(mode+1)]: aug_data,
                     ops['labels_pl'+str(mode+1)]: batch_label,
	  	     ops['smpws_pl'+str(mode+1)]: batch_smpw,
                     ops['data_select']: mode,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=add_empty(feed_dict, mode, ops))
	test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
	tmp,_ = np.histogram(batch_label,range(22))
	labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))

	for b in xrange(batch_label.shape[0]):
	    _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
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
