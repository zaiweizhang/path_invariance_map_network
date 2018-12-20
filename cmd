CUDA_VISIBLE_DEVICES=1 nohup python train_pc_joint_multi_combinesample_queue.py > trainlog_pc &
CUDA_VISIBLE_DEVICES=0 nohup python train_voxel_joint_multi_v1.py > trainlog_voxel_v1 &
CUDA_VISIBLE_DEVICES=0 nohup python train_voxel_joint_multi_v2.py > trainlog_voxel_v2 &

