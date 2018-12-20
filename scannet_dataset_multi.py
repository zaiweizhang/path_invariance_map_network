import pickle
import os
import sys
import numpy as np
import pc_util
import scene_util
from sklearn.cluster import KMeans

class ScannetDataset():
    def __init__(self, root, npoints=[4096, 8192, 12288], split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        if split == 'train':
            self.data_filename = os.path.join(self.root, 'scannet_%s_100.pickle'%(split))
        else:
            self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
	if split=='train':
	    labelweights = np.zeros(21)
	    for seg in self.semantic_labels_list:
		tmp,_ = np.histogram(seg,range(22))
		labelweights += tmp
	    labelweights = labelweights.astype(np.float32)
	    labelweights = labelweights/np.sum(labelweights)
	    self.labelweights = 1/np.log(1.2+labelweights)
	elif split=='test':
	    self.labelweights = np.ones(21)
            
    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set,axis=0)
	coordmin = np.min(point_set,axis=0)
	smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
	smpmin[2] = coordmin[2]
	smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
	smpsz[2] = coordmax[2]-coordmin[2]
	isvalid = False
	for i in range(10):
	    curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
	    curmin = curcenter-[0.75,0.75,1.5]
	    curmax = curcenter+[0.75,0.75,1.5]
	    curmin[2] = coordmin[2]
	    curmax[2] = coordmax[2]
	    curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
	    cur_point_set = point_set[curchoice,:]
	    cur_semantic_seg = semantic_seg[curchoice]
	    if len(cur_semantic_seg)==0:
		continue
	    mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
	    vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
	    vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
	    isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
	    if isvalid:
		break

        point_set = []
        semantic_seg = []
        masks = []
	choice1 = np.random.choice(len(cur_point_set), self.npoints[0], replace=True)
        choice2 = np.random.choice(len(cur_point_set), self.npoints[1], replace=True)
        choice3 = np.random.choice(len(cur_point_set), self.npoints[2], replace=True)
        pc1 = cur_point_set[choice1,:].copy()
        sem1 = cur_semantic_seg[choice1].copy()
        mask1 = mask[choice1].copy()
        pc2 = cur_point_set[choice2,:].copy()
        sem2 = cur_semantic_seg[choice2].copy()
        mask2 = mask[choice2].copy()
        pc3 = cur_point_set[choice3,:].copy()
        sem3 = cur_semantic_seg[choice3].copy()
        mask3 = mask[choice3].copy()
        
        ### getting samples from other point clouds densities
        ### Optimized based on surface variation (https://lgg.epfl.ch/publications/2003/pauly_2003_MFE.pdf)
        hm = pc_util.sample_multi(np.squeeze(pc3))
        #pc_util.write_ply_color_multic(np.squeeze(pc3), (hm-0.1)/0.9*0.7, "test.ply") ### can be used for visualization
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
        sem3_selected = sem3[curchoicepc3,...].copy()
        mask3_selected = mask3[curchoicepc3,...].copy()
                
        curmin = curcenter-[0.75*0.70,0.75*0.70,1.5*0.70]
	curmax = curcenter+[0.75*0.70,0.75*0.70,1.5*0.70]
	curmin[2] = coordmin[2]
	curmax[2] = coordmax[2]
	curchoicepc3 = np.sum((pc3>=(curmin-0.1))*(pc3<=(curmax+0.1)),axis=1)==3
        pc3_selected_f = pc3[curchoicepc3,...].copy()
        sem3_selected_f = sem3[curchoicepc3,...].copy()
        mask3_selected_f = mask3[curchoicepc3,...].copy()

        data_idx1 = np.random.choice(len(np.squeeze(pc3_selected_f)), self.npoints[0], replace=True)
        data_idx2 = np.random.choice(len(np.squeeze(pc3_selected)), self.npoints[1], replace=True)
        pc1_fromPC3 = pc3_selected_f[data_idx1,:].copy()
        sem1_fromPC3 = sem3_selected_f[data_idx1].copy()
        mask1_fromPC3 = mask3_selected_f[data_idx1].copy()

        pc2_fromPC3 = pc3_selected[data_idx2,:].copy()
        sem2_fromPC3 = sem3_selected[data_idx2].copy()
        mask2_fromPC3 = mask3_selected[data_idx2].copy()

        ### pcII to pcIII
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
        sem2_selected = sem2[curchoicepc2,...].copy()
        mask2_selected = mask2[curchoicepc2,...].copy()
        
        data_idx = np.random.choice(len(np.squeeze(pc2_selected)), self.npoints[0], replace=True)
        pc1_fromPC2 = pc2_selected[data_idx,:].copy()
        sem1_fromPC2 = sem2_selected[data_idx].copy()
        mask1_fromPC2 = mask2_selected[data_idx].copy()
        
        point_set = [pc1_fromPC2, pc1_fromPC3, pc1, pc2_fromPC3, pc2, pc3]
        sems = [sem1_fromPC2, sem1_fromPC3, sem1, sem2_fromPC3, sem2, sem3]
        masks = [self.labelweights[sems[0]]*mask1_fromPC2, self.labelweights[sems[1]]*mask1_fromPC3, self.labelweights[sems[2]]*mask1, self.labelweights[sems[3]]*mask2_fromPC3, self.labelweights[sems[4]]*mask2, self.labelweights[sems[5]]*mask3]
        return point_set, sems, masks
        
    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetWholeScene():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
	if split=='train':
	    labelweights = np.zeros(21)
	    for seg in self.semantic_labels_list:
		tmp,_ = np.histogram(seg,range(22))
		labelweights += tmp
	    labelweights = labelweights.astype(np.float32)
	    labelweights = labelweights/np.sum(labelweights)
	    self.labelweights = 1/np.log(1.2+labelweights)
	elif split=='test':
	    self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini,axis=0)
	coordmin = np.min(point_set_ini,axis=0)
	nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
	point_sets = list()
	semantic_segs = list()
	sample_weights = list()
	isvalid = False
	for i in range(nsubvolume_x):
	    for j in range(nsubvolume_y):
		curmin = coordmin+[i*1.5,j*1.5,0]
		curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
		curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
		cur_point_set = point_set_ini[curchoice,:]
	        cur_semantic_seg = semantic_seg_ini[curchoice]
	        if len(cur_semantic_seg)==0:
		    continue
		mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
		choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
		point_set = cur_point_set[choice,:] # Nx3
		semantic_seg = cur_semantic_seg[choice] # N
		mask = mask[choice]
		if sum(mask)/float(len(mask))<0.01:
		    continue
		sample_weight = self.labelweights[semantic_seg]
		sample_weight *= mask # N
		point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
		semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
		sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
	point_sets = np.concatenate(tuple(point_sets),axis=0)
	semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
	sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)

if __name__=='__main__':
    d = ScannetDatasetWholeScene(root = './data', split='test', npoints=8192)
    labelweights_vox = np.zeros(21)
    for ii in xrange(len(d)):
	print ii
        ps,seg,smpw = d[ii]
        for b in xrange(ps.shape[0]):
    	    _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b,smpw[b,:]>0,:], seg[b,smpw[b,:]>0], res=0.02)
	    tmp,_ = np.histogram(uvlabel,range(22))
	    labelweights_vox += tmp
    print labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    exit()


