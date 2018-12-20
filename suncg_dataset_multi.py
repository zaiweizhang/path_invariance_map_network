import pickle
import os
import os.path
import sys
import numpy as np
import pc_util
import scene_util
import scipy.io as sio
import time
from multiprocessing import Process, Manager
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

NUM_REPE = 6
SUBSET = [0, 1, 3]

### Get the consistent labels across different dimensions
def get_vote_parallel(data, pcndex, batch_data, pred_val, voxeldata1, voxeldata2):
    newpred = []
    for i in range(NUM_REPE): ## 7 point representation
        newpred.append(np.zeros(pred_val[i].shape))
    ### We have 9 representations
    for i in range(batch_data[0].shape[0]):
        result_pred = np.zeros([batch_data[-1].shape[1], NUM_REPE+2])
        result_pred[:,-3] = pred_val[-1][i,:] ## 12288 preds
        result_pred[:,-2] = voxeldata1[i,:] ## voxel preds
        result_pred[:,-1] = voxeldata2[i,:] ## voxel preds
        pc_maps = []
        pc12288_tree = KDTree(batch_data[-1][i,:,:], leaf_size=100)
        for j in range(NUM_REPE-1):
            if j in SUBSET:
                ### Sub sampled pc based on surface varation
                result_pred[:,j] = (j+22)#use large class to remove the contribution
                idx = np.squeeze(pc12288_tree.query(batch_data[j][i,:,:], k=1)[1])
                pc_map = {i:idx[i] for i in range(len(batch_data[j][i,:,:]))}
                result_pred[idx,j] = pred_val[j][i,:]
            else:
                pc_tree = KDTree(batch_data[j][i,:,:], leaf_size=100)
                idx = np.squeeze(pc_tree.query(batch_data[-1][i,:,:],k=1)[1])
                result_pred[:,j] = pred_val[j][i,idx]
                idx = np.squeeze(pc12288_tree.query(batch_data[j][i,:,:], k=1)[1])
                pc_map = {i:idx[i] for i in range(len(batch_data[j][i,:,:]))}
            pc_maps.append(pc_map)

        ### Get the popular vote here
        axis = 1
        u, indices = np.unique(result_pred, return_inverse=True)
        voted_pred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(result_pred.shape), None, np.max(indices) + 1), axis=axis)]
        
        newpred[-1][i,:] = voted_pred
        for j in range(NUM_REPE-1):
            for k in range(len(batch_data[j][i,:,:])):
                newpred[j][i,k] = voted_pred[pc_maps[j][k]]
    data[pcndex] = newpred

class SuncgDataset():
    def __init__(self, root, batch_size=8, npoints=[4096, 8192, 12288], split='train', rep="pc"):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.rep = rep
        self.nrep = [3, 2, 1]
        self.batch_size = batch_size
        self.data_filename = os.path.join(self.root, 'scannet_train_unlabel.pickle')
        with open(self.data_filename,'rb') as fp:
            self.scene_list = pickle.load(fp)

        self.train_idxs = np.arange(0, len(self.scene_list))
        np.random.shuffle(self.train_idxs)
        self.num_batches = len(self.scene_list)/self.batch_size
        self.batch_idx = 0
        self.epoch_idx = 0
	if split=='train':
	    self.labelweights = np.ones(21)

    def get_next(self):
        if self.batch_idx >= self.num_batches:
            self.batch_idx = 0
            np.random.shuffle(self.train_idxs)

        batch_data = []
        mask = []
        for i in range(len(self.npoints)):
            for j in range(self.nrep[i]):
                batch_data.append(np.zeros((self.batch_size, self.npoints[i], 3)))
                mask.append(np.zeros((self.batch_size, self.npoints[i])))
        start_idx = self.batch_idx * self.batch_size
        for i in range(self.batch_size):
            ps, smpw = self.getitem(self.train_idxs[i+start_idx])
            counter = 0
            for j in range(len(self.npoints)):
                for k in range(self.nrep[j]):
                    batch_data[counter][i,...] = ps[counter]
                    mask[counter][i,:] = smpw[counter]

                    ### Add the drop point as training
                    dropout_ratio = np.random.random()*0.875 # 0-0.875
                    drop_idx = np.where(np.random.random((ps[counter].shape[0]))<=dropout_ratio)[0]
	            batch_data[counter][i,drop_idx,:] = batch_data[counter][i,0,:]
	            mask[counter][i,drop_idx] *= 0
                    counter += 1
        self.batch_idx += 1
        return batch_data, mask

    def getitem(self, index):
        point_set = self.scene_list[index]
        coordmax = np.max(point_set,axis=0)
	coordmin = np.min(point_set,axis=0)
	isvalid = False
	curcenter = point_set[np.random.choice(len(point_set),1)[0],:]
        curmin = curcenter-[0.75,0.75,1.5]
	curmax = curcenter+[0.75,0.75,1.5]
	curmin[2] = coordmin[2]
	curmax[2] = coordmax[2]
	curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
	cur_point_set = point_set[curchoice,:]
        mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
        point_set = []
        masks = []
	choice1 = np.random.choice(len(cur_point_set), self.npoints[0], replace=True)
        choice2 = np.random.choice(len(cur_point_set), self.npoints[1], replace=True)
        choice3 = np.random.choice(len(cur_point_set), self.npoints[2], replace=True)
        pc1 = cur_point_set[choice1,:].copy()
        mask1 = mask[choice1].copy()
        pc2 = cur_point_set[choice2,:].copy()
        mask2 = mask[choice2].copy()
        pc3 = cur_point_set[choice3,:].copy()
        mask3 = mask[choice3].copy()

        ### getting samples from other point clouds densities
        ### Optimized based on surface variation (https://lgg.epfl.ch/publications/2003/pauly_2003_MFE.pdf)
        hm = pc_util.sample_multi(np.squeeze(pc3))
        idx = np.argsort(hm)
        ### Take the last 20 points
        sal_points_frompc3 = pc3[idx[-20:], ...]
        kmeans = KMeans(n_clusters=3, random_state=0).fit(sal_points_frompc3)
        maxlabel = np.argmax(np.bincount(kmeans.labels_))
        curcenter = kmeans.cluster_centers_[maxlabel,:]
	curmin = curcenter-[0.75*0.88,0.75*0.88,1.5*0.88]
	curmax = curcenter+[0.75*0.88,0.75*0.88,1.5*0.88]
	curmin[2] = coordmin[2]
	curmax[2] = coordmax[2]
	curchoicepc3 = np.sum((pc3>=(curmin-0.1))*(pc3<=(curmax+0.1)),axis=1)==3
        pc3_selected = pc3[curchoicepc3,...].copy()
        mask3_selected = mask3[curchoicepc3,...].copy()
                
        curmin = curcenter-[0.75*0.70,0.75*0.70,1.5*0.70]
	curmax = curcenter+[0.75*0.70,0.75*0.70,1.5*0.70]
	curmin[2] = coordmin[2]
	curmax[2] = coordmax[2]
	curchoicepc3 = np.sum((pc3>=(curmin-0.1))*(pc3<=(curmax+0.1)),axis=1)==3
        pc3_selected_f = pc3[curchoicepc3,...].copy()
        mask3_selected_f = mask3[curchoicepc3,...].copy()

        data_idx1 = np.random.choice(len(np.squeeze(pc3_selected_f)), self.npoints[0], replace=True)
        data_idx2 = np.random.choice(len(np.squeeze(pc3_selected)), self.npoints[1], replace=True)
        pc1_fromPC3 = pc3_selected_f[data_idx1,:].copy()
        mask1_fromPC3 = mask3_selected_f[data_idx1].copy()

        pc2_fromPC3 = pc3_selected[data_idx2,:].copy()
        mask2_fromPC3 = mask3_selected[data_idx2].copy()

        hm = pc_util.sample_multi(np.squeeze(pc2))
        idx = np.argsort(hm)
        ### Take the last 20 points
        sal_points_frompc2 = pc2[idx[-20:], ...]
        kmeans = KMeans(n_clusters=3, random_state=0).fit(sal_points_frompc2)
        maxlabel = np.argmax(np.bincount(kmeans.labels_))
        curcenter = kmeans.cluster_centers_[maxlabel,:]
	curmin = curcenter-[0.75*0.79,0.75*0.79,1.5*0.79]
	curmax = curcenter+[0.75*0.79,0.75*0.79,1.5*0.79]
	curmin[2] = coordmin[2]
	curmax[2] = coordmax[2]
	curchoicepc2 = np.sum((pc2>=(curmin-0.1))*(pc2<=(curmax+0.1)),axis=1)==3
        pc2_selected = pc2[curchoicepc2,...].copy()
        mask2_selected = mask2[curchoicepc2,...].copy()
        
        data_idx = np.random.choice(len(np.squeeze(pc2_selected)), self.npoints[0], replace=True)
        pc1_fromPC2 = pc2_selected[data_idx,:].copy()
        mask1_fromPC2 = mask2_selected[data_idx].copy()

        point_set = [pc1_fromPC2, pc1_fromPC3, pc1, pc2_fromPC3, pc2, pc3]
        masks = [mask1_fromPC2, mask1_fromPC3, mask1, mask2_fromPC3, mask2, mask3]
        return point_set, masks
        
    def __len__(self):
        return len(self.scene_list)

    def wait_other(self):
        sid = self.rep[5]
        while (not os.path.exists("pc_data"+sid+".mat")):
            pass
        time.sleep(1) ## Wait for data to be written
        inmat = sio.loadmat("pc_data"+sid+".mat") 
        data = inmat['batch_data']
        smpw = inmat['batch_smpw']
        os.remove("pc_data"+sid+".mat")
        return data, smpw
    
    def check_gone(self, batch_data, batch_smpw):
        sio.savemat("pc_data1.mat", {"batch_data":batch_data, "batch_smpw": batch_smpw})
        sio.savemat("pc_data2.mat", {"batch_data":batch_data, "batch_smpw": batch_smpw})
        while (os.path.exists("pc_data1.mat")) or (os.path.exists("pc_data2.mat")) :
            pass
        return

    def get_vote_multi(self, bdata, bpred, vdata, vdata2):
        ### Do multi-threading here to reduce time
        numP = bdata[0].shape[0]
        result = []
        proc = []
        stored = Manager().dict()

        for i in range(numP):
            newbdata = [np.expand_dims(bdata[j][i,...], 0) for j in range(NUM_REPE)]
            newbpred = [np.expand_dims(bpred[j][i,...], 0) for j in range(NUM_REPE)]
            newvdata = np.expand_dims(vdata[i,...], 0)
            newvdata2 = np.expand_dims(vdata2[i,...], 0)
            p = Process(target=get_vote_parallel, args=(stored, i, newbdata, newbpred, newvdata, newvdata2))
            p.start()
            proc.append(p)

        for p in proc:
            p.join()

        for ndex in sorted(stored.keys()):
            result.append(stored[ndex])

        reps = []
        for i in range(NUM_REPE):
            reps.append([])
        for i in range(numP):
            for j in range(NUM_REPE):
                reps[j].append(result[i][j])
        result = [np.concatenate(reps[i], 0) for i in range(NUM_REPE)]
        return result

    def ready(self, batch_data, pred_val, mask, label_weights):
        if "voxel" in self.rep:
            sid = self.rep[5]
            sio.savemat(self.rep+".mat", {"batch_data":batch_data, "pred_val":pred_val})
            while (not os.path.exists("pc"+sid+".mat")):
                pass
            time.sleep(1) ## Wait for data to be written
            newdata = sio.loadmat("pc"+sid+".mat")
            os.remove("pc"+sid+".mat")
            return newdata["batch_data"], newdata["pred_val"], label_weights[newdata["pred_val"].astype(np.int32)]*mask
        elif self.rep == "pc":
            while (not os.path.exists("voxel1.mat")) or (not os.path.exists("voxel2.mat")):
                pass
            time.sleep(1) ## Wait for data to be written
            voxeldata1 = sio.loadmat("voxel1.mat")
            os.remove("voxel1.mat")
            voxeldata2 = sio.loadmat("voxel2.mat")
            os.remove("voxel2.mat")
            newpred = self.get_vote_multi(batch_data, pred_val, voxeldata1["pred_val"], voxeldata2["pred_val"])
            
            ## Save voted data to file
            sio.savemat(self.rep+"1.mat", {"batch_data":voxeldata1["batch_data"], "pred_val":newpred[-1]})
            sio.savemat(self.rep+"2.mat", {"batch_data":voxeldata2["batch_data"], "pred_val":newpred[-1]})
            smpws = []
            counter = 0
            for i in range(len(self.npoints)):
                for j in range(self.nrep[i]):
                    smpws.append(label_weights[newpred[counter].astype(np.int32)]*mask[counter])
                    counter += 1
            return batch_data, newpred, smpws
        else:
            print ("only support voxel or pc right now")
            sys.exit(0)

