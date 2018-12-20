""" Utility functions for processing point clouds.

Heavily borrowed from pointnet2
Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement
import operator

import matplotlib.pyplot as pyplot

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from multiprocessing import Process, Manager

def write_ply_color_multic(points, labels, out_filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    ### Write header here
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex %d\n" % N)
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("end_header\n")
    for i in range(N):
        c = pyplot.cm.hsv(labels[i])
        c = [int(x*255) for x in c]
        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()

def surface_variant_para(stored, pcndex, pc):
    num_neighbour = 10
    pca = PCA()
    kdt = KDTree(pc, leaf_size=100, metric='euclidean')
    ### For each point we get the surface variant
    hm = np.zeros(pc.shape[0])
    idx = kdt.query(pc,k=num_neighbour)[1]
    for i in range(len(idx)):
        data = pc[idx[i],:]
        pca.fit(data)
        lambdas = pca.singular_values_
        hm[i] = lambdas[2]/float(sum(lambdas))
        if np.isnan(hm[i]):
            hm[i] = 0
    ### Normalize the surface variant here
    minv = np.min(hm)
    maxv = np.max(hm)
    if float(maxv - minv) == 0:
        stored[pcndex] = np.ones(hm.shape)
    else:
        stored[pcndex] = (hm-minv)/float(maxv - minv)*0.9+0.1
    
def sample_multi(pc):
    ### Do multi-threading here to reduce time
    numP = 16
    result = []
    proc = []
    stored = Manager().dict()

    chunk = len(pc)//numP
    for i in range(numP):
        newbdata = pc[i*chunk:(i+1)*chunk,...]
        p = Process(target=surface_variant_para, args=(stored, i, newbdata))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()

    for ndex in sorted(stored.keys()):
        result.append(stored[ndex])

    result = np.concatenate(result, 0)
    return result    
        
### Multi-view to point cloud conversion
def DepthToPointCloud(image, label, pred, intrinsic):
    # depth image to point cloud
    h, w = image.shape[0], image.shape[1]
    ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
    vals = image[ys, xs]
    labels = label[ys, xs]
    preds = pred[ys, xs]
    valid = (vals != 0)
    ys, xs, vals, labels, preds = ys[valid], xs[valid], vals[valid], labels[valid], preds[valid]
    points = np.zeros([len(ys), 3])
    points[:,0] = (xs-w/2.0) / intrinsic[0] * vals
    points[:,1] = (ys-h/2.0) / intrinsic[0] * vals
    points[:,2] = vals
    return points, labels, preds

def mv_to_pc(batch_image, batch_label, pred_label, batch_pose, intrinsic):
    pcall = []
    labelall = []
    predall = []
    R_base = np.linalg.inv(batch_pose[0])
    for i in range(len(batch_image)):
        pc, label, pred = DepthToPointCloud(np.squeeze(batch_image[i][:,:,0]), np.squeeze(batch_label[i]), np.squeeze(pred_label[i]), intrinsic)
        R = np.matmul(R_base, batch_pose[i])
        pc = np.matmul(R[:3,:3], pc.T)+R[:3,3:4]
        pcall.append(pc.copy())
        labelall.append(label)
        predall.append(pred)
    pcall = (np.concatenate(pcall, 1)).T
    labelall = np.concatenate(labelall, 0)
    predall = np.concatenate(predall, 0)
    pc2obj(pcall[::100,:].T)
    return pcall[::100,:], labelall[::100], predall[::100]

def voting_pc(pc, pred):
    pc_smaller = pc[::10,:]
    pred_smaller = pred[::10]
    pc_reduced = pc[::100,:]
    newpc_dict = {i:{} for i in range(pc_reduced.shape[0])}
    for i in range(len(pc_smaller)):
        if i % 1000 == 0:
            print ("done with pc:", i)
        dist2 = np.sum((pc_reduced - pc_smaller[i,:])**2, axis=1)
        idx = np.argmin(dist2)
        if pred_smaller[i] in newpc_dict[idx].keys():
            newpc_dict[idx][pred_smaller[i]] += 1
        else:
            newpc_dict[idx][pred_smaller[i]] = 1
    newpred = []
    for i in range(len(pc_reduced)):
        newpred.append(max(newpc_dict[i].iteritems(), key=operator.itemgetter(1))[0])
    return np.array(newpred)
        
def mv_to_pc_voting(batch_image, batch_label, pred_label, batch_pose, intrinsic):
    pcall = []
    labelall = []
    predall = []
    R_base = np.linalg.inv(batch_pose[0])
    for i in range(len(batch_image)):
        pc, label, pred = DepthToPointCloud(np.squeeze(batch_image[i][:,:,0]), np.squeeze(batch_label[i]), np.squeeze(pred_label[i]), intrinsic)
        R = np.matmul(R_base, batch_pose[i])
        pc = np.matmul(R[:3,:3], pc.T)+R[:3,3:4]
        pcall.append(pc.copy())
        labelall.append(label)
        predall.append(pred)
    pcall = (np.concatenate(pcall, 1)).T
    labelall = np.concatenate(labelall, 0)
    predall = np.concatenate(predall, 0)
    pred_reduced = voting_pc(pcall, predall)
    pc2obj(pcall[::100,:].T)
    return pcall[::100,:], labelall[::100], pred_reduced

def pc2obj(pc, filepath='test.obj'):
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------
def point_cloud_label_to_surface_voxel_label(point_cloud, label, res=0.0484):
    coordmax = np.max(point_cloud,axis=0)
    coordmin = np.min(point_cloud,axis=0)
    nvox = np.ceil((coordmax-coordmin)/res)
    vidx = np.ceil((point_cloud-coordmin)/res)
    vidx = vidx[:,0]+vidx[:,1]*nvox[0]+vidx[:,2]*nvox[0]*nvox[1]
    uvidx = np.unique(vidx)
    if label.ndim==1:
        uvlabel = [np.argmax(np.bincount(label[vidx==uv].astype(np.uint32))) for uv in uvidx]
    else:
        assert(label.ndim==2)
	uvlabel = np.zeros(len(uvidx),label.shape[1])
	for i in range(label.shape[1]):
	    uvlabel[:,i] = np.array([np.argmax(np.bincount(label[vidx==uv,i].astype(np.uint32))) for uv in uvidx])
    return uvidx, uvlabel, nvox

def point_cloud_label_to_surface_voxel_label_fast(point_cloud, label, res=0.0484):
    coordmax = np.max(point_cloud,axis=0)
    coordmin = np.min(point_cloud,axis=0)
    nvox = np.ceil((coordmax-coordmin)/res)
    vidx = np.ceil((point_cloud-coordmin)/res)
    vidx = vidx[:,0]+vidx[:,1]*nvox[0]+vidx[:,2]*nvox[0]*nvox[1]
    uvidx, vpidx = np.unique(vidx,return_index=True)
    if label.ndim==1:
        uvlabel = label[vpidx]
    else:
        assert(label.ndim==2)
	uvlabel = label[vpidx,:]
    return uvidx, uvlabel, nvox

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.1, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)

def point_cloud_label_to_volume_batch(point_clouds, labels, weights, vsize=12, radius=1.1, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    label_list = []
    weight_list = []
    for b in range(point_clouds.shape[0]):
        vol, label, weight = point_cloud_label_to_volume(np.squeeze(point_clouds[b,:,:]), np.squeeze(labels[b,:]), np.squeeze(weights[b,:]), vsize, radius)
        vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
        label_list.append(np.expand_dims(label, 0))
        weight_list.append(np.expand_dims(weight, 0))
    
    return np.concatenate(vol_list, 0), np.concatenate(label_list, 0), np.concatenate(weight_list, 0)

def point_cloud_label_to_volume_batch_exact(point_clouds, vsize=12, radius=1.1, flatten=True): 
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = np.zeros((vsize,vsize,vsize))
        voxel = 2*radius/float(vsize)
        locations = (np.squeeze(point_clouds[b,:,:]) + radius)/voxel
        locations = locations.astype(int)
        vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
        vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    return np.concatenate(vol_list, 0)

def point_cloud_label_to_volume(points, label, weight, vsize, radius=1.1):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    la = np.zeros((vsize,vsize,vsize))
    we = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    la[locations[:,0],locations[:,1],locations[:,2]] = label
    we[locations[:,0],locations[:,1],locations[:,2]] = weight
    return vol, la, we

def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

def volume_topc_batch(pred_val, batch_label_vol, batch_smpw_vol):
    bsize = pred_val.shape[0]
    pred_pc = []
    label_pc = []
    smpw_pc = []
    other_pc = []
    aug_data = []
    vsize = pred_val.shape[1]
    for i in range(bsize):
        points = []
        points_label = []
        points_smpw = []
        points_other = []
        points_aug = []
        for a in range(vsize):
            for b in range(vsize):
                for c in range(vsize):
                    if batch_label_vol[i,a,b,c] > 0:
                        points.append(pred_val[i,a,b,c])
                        points_label.append(batch_label_vol[i,a,b,c])
                        points_smpw.append(batch_smpw_vol[i,a,b,c])
                        points_aug.append(np.array([a,b,c]))
                    elif batch_label_vol[i,a,b,c] == 0 and batch_smpw_vol[i,a,b,c] > 0:
                        points_other.append(pred_val[i,a,b,c])
        if len(points) == 0:
            continue

        pred_pc.append(np.array(points))
        label_pc.append(np.array(points_label))
        smpw_pc.append(np.array(points_smpw))
        other_pc.append(np.array(points_other))
        aug_data.append(np.array(points_aug))
    return pred_pc, label_pc, smpw_pc, other_pc, aug_data

def volume_topc_batch_exact(pred_val, batch_data, radius=1.1, vsize=32): 
    bsize = pred_val.shape[0]
    pred_pc = []
    label_pc = []
    for i in range(bsize):
        cur_data = batch_data[i,:,:]
        cur_val = pred_val[i,:,:,:]

        voxel = 2*radius/float(vsize)
        cur_data = (np.squeeze(cur_data) + radius)/voxel
        cur_data = cur_data.astype(int)

        points_label = cur_val[cur_data[:,0], cur_data[:,1], cur_data[:,2]]
        pred_pc.append(points_label)
    return np.array(pred_pc)

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b,:,:], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)

def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize,vsize,vsize,num_sample,3))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i,j,k) not in loc2pc:
                    vol[i,j,k,:,:] = np.zeros((num_sample,3))
                else:
                    pc = loc2pc[(i,j,k)] # a list of (3,) arrays
                    pc = np.vstack(pc) # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0]>num_sample:
                        choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                        pc = pc[choices,:]
                    elif pc.shape[0]<num_sample:
                        pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i,j,k])+0.5)*voxel - radius
                    #print 'pc center: ', pc_center
                    pc = (pc - pc_center) / voxel # shift and scale
                    vol[i,j,k,:,:] = pc 
                #print (i,j,k), vol[i,j,k,:,:]
    return vol

def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b,:,:], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2*radius/float(imgsize)
    locations = (points[:,0:2] + radius)/pixel # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i,j) not in loc2pc:
                img[i,j,:,:] = np.zeros((num_sample,3))
            else:
                pc = loc2pc[(i,j)]
                pc = np.vstack(pc)
                if pc.shape[0]>num_sample:
                    choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                    pc = pc[choices,:]
                elif pc.shape[0]<num_sample:
                    pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                pc_center = (np.array([i,j])+0.5)*pixel - radius
                pc[:,0:2] = (pc[:,0:2] - pc_center)/pixel
                img[i,j,:,:] = pc
    return img
# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])
       
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
    image = image / np.max(image)
    return image

def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """ 
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
    img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
    img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    from PIL import Image
    points = read_ply('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save('piano.jpg')

if __name__=="__main__":
    point_cloud_three_views_demo()


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)

def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    fout = open(out_filename, 'w')
    colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()
