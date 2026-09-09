import os
from os.path import join
import gzip, json, pickle
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

import cv2
import multiprocessing
import argparse
from skimage.draw import line
from matplotlib.path import Path

# All data in the Bench2Drive dataset are in the left-handed coordinate system.
# This code converts all coordinate systems (world coordinate system, vehicle coordinate system, camera coordinate system, and lidar coordinate system) to the right-handed coordinate system
# consistent with the nuscenes dataset.

DATAROOT = '/home/fqz02/Works/WangRunhu/demo02/navsim_workspace/dataset/Bench2Drive/'
MAP_ROOT = 'data/bench2drive/maps'
OUT_DIR = '/home/fqz02/Works/WangRunhu/demo02/navsim_workspace/dataset/Bench2Drive/data/infos'

max_agents = 16
lidar_min_x: float = -32
lidar_max_x: float = 32
lidar_min_y: float = -32
lidar_max_y: float = 32

MAX_DISTANCE = 75              # Filter bounding boxes that are too far from the vehicle
FILTER_Z_SHRESHOLD = 10        # Filter bounding boxes that are too high/low from the vehicle
FILTER_INVISINLE = True        # Filter bounding boxes based on visibility
NUM_VISIBLE_SHRESHOLD = 1      # Filter bounding boxes with fewer visible vertices than this value
NUM_OUTPOINT_SHRESHOLD = 7     # Filter bounding boxes where the number of vertices outside the frame is greater than this value in all cameras

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
CAMERA_TO_FOLDER_MAP = {'CAM_FRONT':'rgb_front', 'CAM_FRONT_LEFT':'rgb_front_left', 'CAM_FRONT_RIGHT':'rgb_front_right', 'CAM_BACK':'rgb_back', 'CAM_BACK_LEFT':'rgb_back_left', 'CAM_BACK_RIGHT':'rgb_back_right'}

stand_to_ue4_rotate = np.array([[ 0, 0, 1, 0],
                                [ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 0, 1]])

lidar_to_righthand_ego = np.array([[  0, 1, 0, 0],
                                   [ -1, 0, 0, 0],
                                   [  0, 0, 1, 0],
                                   [  0, 0, 0, 1]])

lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1,1] = -1

def apply_trans(vec,world2ego):
    vec = np.concatenate((vec,np.array([1])))
    t = world2ego @ vec
    return t[0:3]

def get_pose_matrix(dic):
    new_matrix = np.zeros((4,4))
    new_matrix[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=dic['theta']-np.pi/2).rotation_matrix
    new_matrix[0,3] = dic['x']
    new_matrix[1,3] = dic['y']
    new_matrix[3,3] = 1
    return new_matrix

def get_npc2world(npc):
    for key in ['world2vehicle','world2ego','world2sign','world2ped']:
        if key in npc.keys():
            npc2world = np.linalg.inv(np.array(npc[key]))
            yaw_from_matrix = np.arctan2(npc2world[1,0], npc2world[0,0])
            yaw = npc['rotation'][-1] / 180 * np.pi
            if abs(yaw-yaw_from_matrix)> 0.01:
                npc2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
            npc2world = left2right @ npc2world @ left2right
            return npc2world
    npc2world = np.eye(4)
    npc2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=npc['rotation'][-1]/180*np.pi).rotation_matrix
    npc2world[0:3,3] = np.array(npc['location'])
    return left2right @ npc2world @ left2right


def get_global_trigger_vertex(center,extent,yaw_in_degree):
    x,y = center[0],-center[1]
    dx,dy = extent[0],extent[1]
    yaw_in_radians = -yaw_in_degree/180*np.pi
    vertex_in_self = np.array([[ dx, dy],
                               [-dx, dy],
                               [-dx,-dy],
                               [ dx,-dy]])
    rotate_matrix = np.array([[np.cos(yaw_in_radians),-np.sin(yaw_in_radians)],
                              [np.sin(yaw_in_radians), np.cos(yaw_in_radians)]])
    rotated_vertex = (rotate_matrix @ vertex_in_self.T).T
    vertex_in_global = np.array([[x,y]]).repeat(4,axis=0) + rotated_vertex
    return vertex_in_global



def get_image_point(loc, K, w2c):
    point = np.array([loc[0], loc[1], loc[2], 1])
    point_camera = np.dot(w2c, point)
    point_camera = point_camera[0:3]
    depth = point_camera[2]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], depth

def get_action(index):
	Discrete_Actions_DICT = {
		0:  (0, 0, 1, False),
		1:  (0.7, -0.5, 0, False),
		2:  (0.7, -0.3, 0, False),
		3:  (0.7, -0.2, 0, False),
		4:  (0.7, -0.1, 0, False),
		5:  (0.7, 0, 0, False),
		6:  (0.7, 0.1, 0, False),
		7:  (0.7, 0.2, 0, False),
		8:  (0.7, 0.3, 0, False),
		9:  (0.7, 0.5, 0, False),
		10: (0.3, -0.7, 0, False),
		11: (0.3, -0.5, 0, False),
		12: (0.3, -0.3, 0, False),
		13: (0.3, -0.2, 0, False),
		14: (0.3, -0.1, 0, False),
		15: (0.3, 0, 0, False),
		16: (0.3, 0.1, 0, False),
		17: (0.3, 0.2, 0, False),
		18: (0.3, 0.3, 0, False),
		19: (0.3, 0.5, 0, False),
		20: (0.3, 0.7, 0, False),
		21: (0, -1, 0, False),
		22: (0, -0.6, 0, False),
		23: (0, -0.3, 0, False),
		24: (0, -0.1, 0, False),
		25: (1, 0, 0, False),
		26: (0, 0.1, 0, False),
		27: (0, 0.3, 0, False),
		28: (0, 0.6, 0, False),
		29: (0, 1.0, 0, False),
		30: (0.5, -0.5, 0, True),
		31: (0.5, -0.3, 0, True),
		32: (0.5, -0.2, 0, True),
		33: (0.5, -0.1, 0, True),
		34: (0.5, 0, 0, True),
		35: (0.5, 0.1, 0, True),
		36: (0.5, 0.2, 0, True),
		37: (0.5, 0.3, 0, True),
		38: (0.5, 0.5, 0, True),
		}
	throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
	return throttle, steer, brake


def gengrate_map(map_root):
    map_infos = {}
    for file_name in os.listdir(map_root):
        if '.npz' in file_name:
            map_info = dict(np.load(join(map_root,file_name), allow_pickle=True)['arr'])
            town_name = file_name.split('_')[0]
            map_infos[town_name] = {}
            lane_points = []
            lane_types = []
            lane_sample_points = []
            trigger_volumes_points = []
            trigger_volumes_types = []
            trigger_volumes_sample_points = []
            for road_id, road in map_info.items():
                for lane_id, lane in road.items():
                    if lane_id == 'Trigger_Volumes':
                        for single_trigger_volume in lane:
                            points = np.array(single_trigger_volume['Points'])
                            points[:,1] *= -1 #left2right
                            trigger_volumes_points.append(points)
                            trigger_volumes_sample_points.append(points.mean(axis=0))
                            trigger_volumes_types.append(single_trigger_volume['Type'])
                    else:
                        for single_lane in lane:
                            points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                            points[:,1] *= -1
                            lane_points.append(points)
                            lane_types.append(single_lane['Type'])
                            lane_lenth = points.shape[0]
                            if lane_lenth % 50 != 0:
                                devide_points = [50*i for i in range(lane_lenth//50+1)]
                            else:
                                devide_points = [50*i for i in range(lane_lenth//50)]
                            devide_points.append(lane_lenth-1)
                            lane_sample_points_tmp = points[devide_points]
                            lane_sample_points.append(lane_sample_points_tmp)
            map_infos[town_name]['lane_points'] = lane_points
            map_infos[town_name]['lane_sample_points'] = lane_sample_points
            map_infos[town_name]['lane_types'] = lane_types
            map_infos[town_name]['trigger_volumes_points'] = trigger_volumes_points
            map_infos[town_name]['trigger_volumes_sample_points'] = trigger_volumes_sample_points
            map_infos[town_name]['trigger_volumes_types'] = trigger_volumes_types
    with open(join(OUT_DIR,'b2d_map_infos.pkl'),'wb') as f:
        pickle.dump(map_infos,f)

# lw 将 next_command 转换为独热编码
def cmd2onehot(command, num_classes):
    if command is None or not isinstance(command, int) or command < 1 or command > num_classes:
        return [0] * num_classes  # 对于无效值，返回全零向量
    one_hot = [0] * num_classes
    one_hot[command - 1] = 1  # 因为 next_command 的值是 1-6，而索引是 0-5
    return one_hot
'''
VOID = -1
LEFT = 1
RIGHT = 2
STRAIGHT = 3
LANEFOLLOW = 4
CHANGELANELEFT = 5
CHANGELANERIGHT = 6
'''

# def generate_bev_semantic(frame_data, map_infos):
#     # 初始化BEV画布
#     bev_map = np.zeros((256, 128), dtype=np.uint8)  # 注意行列顺序为(height, width)
    
#     # 获取地图信息
#     town_name = frame_data['town_name']
#     map_info = map_infos.get(town_name, {})
    
#     # 自车坐标变换参数
#     world2ego = frame_data['world2ego']
#     lidar_min_x, lidar_max_x = -32, 32
#     lidar_min_y, lidar_max_y = -32, 32
    
#     # 转换函数：全局坐标到自车坐标系
#     def global_to_ego(p_global):
#         p_homo = np.array([p_global[0], p_global[1], 0, 1])
#         p_ego = world2ego @ p_homo
#         return p_ego[:2]
    
#     # 处理车道线和交叉路口
#     lane_points = map_info.get('lane_points', [])
#     lane_types = map_info.get('lane_types', [])
#     for lane_idx, points in enumerate(lane_points):
#         lane_type = lane_types[lane_idx]
#         transformed = []
#         for p in points:
#             p_ego = global_to_ego(p)
#             if (lidar_min_x <= p_ego[0] <= lidar_max_x) and (lidar_min_y <= p_ego[1] <= lidar_max_y):
#                 # 转换为像素坐标
#                 x = int((p_ego[0] - lidar_min_x) / (lidar_max_x - lidar_min_x) * 127)
#                 y = int((lidar_max_y - p_ego[1]) / (lidar_max_y - lidar_min_y) * 255)
#                 transformed.append((x, y))
#         # 绘制线段
#         for i in range(len(transformed)-1):
#             start, end = transformed[i], transformed[i+1]
#             rr, cc = line(start[1], start[0], end[1], end[0])  # 注意行列顺序
#             valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 128)
#             bev_map[rr[valid], cc[valid]] = 2 if lane_type == 'Intersection' else 1
    
#     # 处理触发区域
#     trigger_volumes = map_info.get('trigger_volumes_points', [])
#     trigger_types = map_info.get('trigger_volumes_types', [])
#     for vol_idx, points in enumerate(trigger_volumes):
#         vol_type = trigger_types[vol_idx]
#         transformed = []
#         for p in points:
#             p_ego = global_to_ego(p)
#             if (lidar_min_x <= p_ego[0] <= lidar_max_x) and (lidar_min_y <= p_ego[1] <= lidar_max_y):
#                 x = int((p_ego[0] - lidar_min_x) / (lidar_max_x - lidar_min_x) * 127)
#                 y = int((lidar_max_y - p_ego[1]) / (lidar_max_y - lidar_min_y) * 255)
#                 transformed.append((x, y))
#         if len(transformed) >= 3:
#             polygon = Path(transformed)
#             x, y = np.meshgrid(np.arange(128), np.arange(256))
#             points = np.vstack((x.ravel(), y.ravel())).T
#             mask = polygon.contains_points(points).reshape(256, 128)
#             if vol_type == 'SideWalk':
#                 bev_map[mask] = 3
#             elif vol_type == 'Parking':
#                 bev_map[mask] = 4
    
#     # 处理动态物体
#     agent_status = frame_data['agent_status']
#     agent_types = frame_data['agent_types']
#     for i in range(max_agents):
#         if agent_types[i] == 0:  # 无效agent跳过
#             continue
#         if not frame_data['agent_labels'][i]:
#             continue
#         agent = agent_status[i]
#         # 类型编码转换（1:车辆->5, 2:行人->6）
#         class_id = 5 if agent_types[i] == 1 else 6
        
#         # 获取参数
#         x_center, y_center, heading, width, length = agent[0], agent[1], agent[2], agent[3], agent[4]
#         # 计算顶点
#         half_w, half_l = width/2, length/2
#         rot = np.array([[np.cos(heading), -np.sin(heading)],
#                         [np.sin(heading), np.cos(heading)]])
#         corners = np.array([[half_l, half_w], [half_l, -half_w],
#                             [-half_l, -half_w], [-half_l, half_w]])
#         rotated = np.dot(corners, rot.T) + np.array([x_center, y_center])
#         # 转换到像素坐标
#         pixel_corners = []
#         for p in rotated:
#             x, y = p[0], p[1]
#             if (lidar_min_x <= x <= lidar_max_x) and (lidar_min_y <= y <= lidar_max_y):
#                 px = int((x - lidar_min_x) / 64 * 127)
#                 py = int((lidar_max_y - y) / 64 * 255)
#                 pixel_corners.append((px, py))
#         # 绘制多边形
#         if len(pixel_corners) >= 3:
#             polygon = Path(pixel_corners)
#             x, y = np.meshgrid(np.arange(128), np.arange(256))
#             points = np.vstack((x.ravel(), y.ravel())).T
#             mask = polygon.contains_points(points).reshape(256, 128)
#             bev_map[mask] = class_id
    
#     return bev_map

def preprocess(folder_list,index,tmp_dir,train_or_val):

    data_root = DATAROOT
    cameras = CAMERAS
    final_data = []
    future_frames = 8*5

    map_file = '/home/fqz02/Works/WangRunhu/demo02/navsim_workspace/dataset/Bench2Drive/b2d_map_infos.pkl'
    with open(map_file,'rb') as f: 
        map_infos = pickle.load(f)

    if index == 0:
        folders = tqdm(folder_list)
    else:
        folders = folder_list

    for folder_name in folders:
        folder_path = join(data_root, folder_name)
        last_position_dict = {}
        seq_x = []
        seq_y = []
        seq_z = []
        seq_yaw = []
        seq_future_x = []
        seq_future_y = []
        seq_future_z = []

        # 获取城镇名称
        # town_name = folder_name.split('/')[1].split('_')[1]

        length = len([name for name in os.listdir(os.path.join(folder_path,'anno'))]) - 1 # drop last frame
        for ann_name in sorted(os.listdir(join(folder_path,'anno')),key= lambda x: int(x.split('.')[0])):
            # i = int(ann_name.split('.')[0])            # ann_name = '00000.json.gz'
            with gzip.open(join(folder_path,'anno',ann_name), 'rt', encoding='utf-8') as gz_file:
                anno = json.load(gz_file)
            seq_x.append(anno['x'])
            seq_y.append(anno['y'])
            world2lidar_z = lefthand_ego_to_lidar @ np.array(anno['sensors']['LIDAR_TOP']['lidar2ego']) @ left2right @ lidar_to_righthand_ego
            for npc_z in anno['bounding_boxes']:
                if npc_z['class'] !='ego_vehicle':continue
                center_z = np.array([npc_z['center'][0],-npc_z['center'][1],npc_z['center'][2]])
                local_center_z = apply_trans(center_z,world2lidar_z)
                ego_z = local_center_z[2]
            seq_z.append(ego_z)
            yaw = -np.nan_to_num(anno['theta'],nan=np.pi)+np.pi/2  
            seq_yaw.append(yaw)

        for i in range(0,length-future_frames-5):
            ann_name = f'{i:05}.json.gz'
            with gzip.open(join(folder_path,'anno',ann_name), 'rt', encoding='utf-8') as gz_file:
                anno = json.load(gz_file)
            seq_future_x.append(seq_x[i+5:i+future_frames+5:5])
            seq_future_y.append(seq_y[i+5:i+future_frames+5:5])
            seq_future_z.append(seq_z[i+5:i+future_frames+5:5])

        for ann_idx in range(0,length-future_frames-5):
        # for ann_name in sorted(os.listdir(join(folder_path,'anno')),key= lambda x: int(x.split('.')[0])):
            ann_name = f'{ann_idx:05}.json.gz'
            position_dict = {}
            frame_data = {}
            # cam_gray_depth = {}

            with gzip.open(join(folder_path,'anno',ann_name), 'rt', encoding='utf-8') as gz_file:
                anno = json.load(gz_file) 
            
            frame_data['folder'] = folder_name
            frame_data['town_name'] =  folder_name.split('_')[1]
            # frame_data['command_far_xy'] = np.array([anno['x_command_far'],-anno['y_command_far']])
            # frame_data['command_far'] = anno['command_far']
            # frame_data['command_near_xy'] = np.array([anno['x_command_near'],-anno['y_command_near']])
            # frame_data['command_near'] = anno['command_near']
            # frame_data['frame_idx'] = int(ann_name.split('.')[0])
            ego_yaw = -np.nan_to_num(anno['theta'],nan=np.pi)+np.pi/2  
            # frame_data['ego_yaw'] = ego_yaw
            # frame_data['ego_translation'] = np.array([anno['x'],-anno['y'],0])  # (x,y,z)

            # lw
            ego_x = anno['x']
            ego_y = anno['y']
            ego_z = seq_z[ann_idx]

            # frame_data['ego_x'] = ego_x
            # frame_data['ego_y'] = ego_y
            waypoints = []
            for i in range(8):
                # 构建旋转矩阵
                R = np.array([
                    [np.cos(ego_yaw), np.sin(ego_yaw)],
                    [-np.sin(ego_yaw), np.cos(ego_yaw)]
                ])

                # 计算相对位置
                local_command_point = np.array([seq_future_x[ann_idx][i]-ego_x, seq_future_y[ann_idx][i]-ego_y])
                local_command_point = R.dot(local_command_point)
                dz = seq_future_z[ann_idx][i] - ego_z
                waypoints.append([local_command_point[0], local_command_point[1],dz])

            frame_data['waypoints'] = np.array(waypoints)
            cmd_num_classes = 6 
            next_command = anno['next_command']
            next_command_onehot = cmd2onehot(next_command,cmd_num_classes)
            frame_data['cmd_onehot'] = np.array(next_command_onehot)
            speed = anno['speed'] 
            speed_x = speed * np.cos(yaw)
            speed_y = speed * np.sin(yaw)
            frame_data['ego_vel'] = np.array([speed_x,speed_y])               # speed 2
            frame_data['ego_accel'] = np.array([anno['acceleration'][0],-anno['acceleration'][1]]) # (x,y,z)三个轴加速度
            # frame_data['ego_vel'] = np.array([anno['speed'],0,0])               # speed 3
            # frame_data['ego_accel'] = np.array([anno['acceleration'][0],-anno['acceleration'][1],anno['acceleration'][2]]) # (x,y,z)三个轴加速度

            # frame_data['ego_rotation_rate'] = -np.array(anno['angular_velocity'])
            # frame_data['ego_size'] = np.array([anno['bounding_boxes'][0]['extent'][1],anno['bounding_boxes'][0]['extent'][0],anno['bounding_boxes'][0]['extent'][2]])*2
            world2ego = left2right @ anno['bounding_boxes'][0]['world2ego'] @ left2right
            # frame_data['world2ego'] = world2ego
            # if frame_data['frame_idx'] == 0:
            #     expert_file_path = join(folder_path,'expert_assessment','-0001.npz')
            # else:
            #     expert_file_path = join(folder_path,'expert_assessment',str(frame_data['frame_idx']-1).zfill(5)+'.npz')
            # expert_data = np.load(expert_file_path,allow_pickle=True)['arr_0']
            # action_id = expert_data[-1]
            # value = expert_data[-2]
            # expert_feature = expert_data[:-2]
            # throttle, steer, brake = get_action(action_id)
            # frame_data['brake'] = brake
            # frame_data['throttle'] = throttle
            # frame_data['steer'] = steer
            #frame_data['action_id'] = action_id
            #frame_data['value'] = value
            #frame_data['expert_feature'] = expert_feature

            ###get sensor infos###
            sensor_infos = {}
            # for cam in CAMERAS:
            #     sensor_infos[cam] = {}
            #     # sensor_infos[cam]['cam2ego'] = left2right @ np.array(anno['sensors'][cam]['cam2ego']) @ stand_to_ue4_rotate 
            #     # sensor_infos[cam]['intrinsic'] = np.array(anno['sensors'][cam]['intrinsic'])
            #     # sensor_infos[cam]['world2cam'] = np.linalg.inv(stand_to_ue4_rotate) @ np.array(anno['sensors'][cam]['world2cam']) @left2right
            #     # sensor_infos[cam]['data_path'] = join(folder_name,'camera',CAMERA_TO_FOLDER_MAP[cam],ann_name.split('.')[0]+'.jpg')
            # #     # 深度相机
            #     # cam_gray_depth[cam] = cv2.imread(join(data_root,sensor_infos[cam]['data_path']).replace('rgb_','depth_').replace('.jpg','.png'))[:,:,0]
            #  # lw
            sensor_infos['cam_data_path'] = join(data_root,folder_name,'cam',ann_name.split('.')[0]+'.jpg')
            # sensor_infos['LIDAR_TOP'] = {}
            # sensor_infos['LIDAR_TOP']['lidar2ego'] = left2right @ np.array(anno['sensors']['LIDAR_TOP']['lidar2ego']) @ left2right @ lidar_to_righthand_ego
            world2lidar = lefthand_ego_to_lidar @ np.array(anno['sensors']['LIDAR_TOP']['world2lidar']) @ left2right
            # sensor_infos['LIDAR_TOP']['world2lidar'] = world2lidar
            sensor_infos['lidar_2d_npy_path'] = join(data_root,folder_name,'lidar_npy',ann_name.split('.')[0]+'.npy')
            frame_data['sensors'] = sensor_infos

            ###get bounding_boxes infos###
            # gt_boxes = []
            # gt_names = []
            # gt_ids = []
            agent_status = np.zeros((max_agents, 5), dtype=np.float32)
            agent_labels = np.zeros(max_agents, dtype=bool)

            tmp_agent_distance = []
            tmp_agent_status = []
            tmp_agent_labels = []

            for npc in anno['bounding_boxes']:
                # 过滤
                if npc['class'] == 'ego_vehicle': continue
                if npc['class'] in ['traffic_light', 'traffic_sign']: continue
                if npc['distance'] > MAX_DISTANCE: continue
                if abs(npc['location'][2] - anno['bounding_boxes'][0]['location'][2]) > FILTER_Z_SHRESHOLD: continue

                center = np.array([npc['center'][0],-npc['center'][1],npc['center'][2]]) # left hand -> right hand
                extent = np.array([npc['extent'][1],npc['extent'][0],npc['extent'][2]])  # lwh -> wlh
                position_dict[npc['id']] = center

                # 使用 world2lidar 变换矩阵将 center 从世界坐标系转换到 LiDAR 坐标系。
                local_center = apply_trans(center, world2lidar)
                size = extent * 2   # 宽 高

                # 计算朝向
                if 'world2vehicle' in npc.keys():
                    # left2right = np.eye(4) 
                    world2vehicle = left2right @ np.array(npc['world2vehicle'])@left2right
                    vehicle2lidar = world2lidar @ np.linalg.inv(world2vehicle) 
                    yaw_local = np.arctan2(vehicle2lidar[1,0], vehicle2lidar[0,0])

                else:
                    yaw_local = -npc['rotation'][-1]/180*np.pi - ego_yaw+np.pi / 2  
                yaw_local_in_lidar_box = -yaw_local - np.pi / 2  
                while yaw_local < -np.pi:
                    yaw_local += 2*np.pi
                while yaw_local > np.pi:
                    yaw_local -= 2*np.pi  

                # 如果目标物体有 speed 属性，则直接使用。否则，根据上一帧的位置计算速度。
                if 'speed' in npc.keys():
                    if 'vehicle' in npc['class']:  # only vehicles have correct speed
                        speed = npc['speed']
                    else:
                        if npc['id'] in last_position_dict.keys():  #calculate speed for other object
                            speed = np.linalg.norm((center-last_position_dict[npc['id']])[0:2]) * 10
                        else:
                            speed = 0
                else:
                    speed = 0

                # # 点云数量
                # if 'num_points' in npc.keys():
                #     num_points = npc['num_points']
                # else:
                #     num_points = -1

                # npc2world = get_npc2world(npc)

                # 计算分速度
                speed_x = speed * np.cos(yaw_local)
                speed_y = speed * np.sin(yaw_local)

                # ###fliter_bounding_boxes###
                # if FILTER_INVISINLE:    # true  通过计算目标物体在相机视野中的可见性进行过滤。
                #     valid = False
                #     box2lidar = np.eye(4)
                #     box2lidar[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=yaw_local).rotation_matrix
                #     box2lidar[0:3,3] = local_center
                #     lidar2box = np.linalg.inv(box2lidar)
                #     raw_verts = calculate_cube_vertices(local_center,extent)
                #     verts = []
                #     for raw_vert in raw_verts:
                #         tmp = np.dot(lidar2box, [raw_vert[0], raw_vert[1], raw_vert[2],1])
                #         tmp[0:3] += local_center
                #         verts.append(tmp.tolist()[:-1])
                #     for cam in cameras:
                #         lidar2cam = np.linalg.inv(frame_data['sensors'][cam]['cam2ego']) @ sensor_infos['LIDAR_TOP']['lidar2ego']
                #         test_points = [] 
                #         test_depth = []
                #         for vert in verts:
                #             point, depth = get_image_point(vert, frame_data['sensors'][cam]['intrinsic'], lidar2cam)
                #             if depth > 0:
                #                 test_points.append(point)
                #                 test_depth.append(depth)

                #         num_visible_vertices, num_invisible_vertices, num_vertices_outside_camera, colored_points = calculate_occlusion_stats(np.array(test_points), np.array(test_depth),  cam_gray_depth[cam], max_render_depth=MAX_DISTANCE)
                #         if num_visible_vertices>NUM_VISIBLE_SHRESHOLD and num_vertices_outside_camera<NUM_OUTPOINT_SHRESHOLD:
                #             valid = True
                #             break
                # else:
                #     valid = True
                # if valid:   # false      
                #     gt_boxes.append(np.concatenate([local_center,size,np.array([yaw_local_in_lidar_box,speed_x,speed_y])]))
                #     gt_names.append(npc['type_id'])
                #     gt_ids.append(int(npc['id']))

                box_x = local_center[0]
                box_y = local_center[1]
                box_heading = yaw_local
                box_width = size[0]  # width
                box_length = size[1]  # length
                # obj_type = 1 if npc['class'] == "walker" else 0
                bounding_box_data = np.array([box_x, box_y, box_heading, box_width, box_length])
                # 同时接受vehicle和walker类别
                if npc['class'] in ["vehicle", "walker"]:
                    # 可选：添加额外的过滤条件（例如对车辆做LiDAR范围检查）
                    # if npc['class'] == "vehicle":
                    if not _xy_in_lidar(box_x, box_y):  # 车辆需要LiDAR范围检查
                        continue
                    # walker类不做LiDAR范围检查
                    tmp_agent_status.append(bounding_box_data)
                    tmp_agent_labels.append(True)
                    tmp_agent_distance.append(npc['distance'])  # 使用原始距离数据
                    # tmp_agent_types.append(1 if npc['class'] == "vehicle" else 2)  # 车辆为1，行人为2

                # 循环结束后处理临时数据
            if len(tmp_agent_status) > 0:
                # 转换为numpy数组
                tmp_agent_status = np.array(tmp_agent_status)  # shape: (N,5)
                tmp_agent_labels = np.array(tmp_agent_labels)  # shape: (N,)
                tmp_agent_distance = np.array(tmp_agent_distance)  # shape: (N,)

                # 按距离排序（从小到大）
                sorted_indices = np.argsort(tmp_agent_distance)
                # 截取前max_agents个
                selected_indices = sorted_indices[:max_agents]
                # 填充到最终容器
                num_valid = min(len(tmp_agent_status), max_agents)
               
                agent_status[:num_valid] = tmp_agent_status[selected_indices]
                agent_labels[:num_valid] = True
                # 在填充agent_states后添加类型保存
                # agent_types = np.full(max_agents, 0, dtype=np.int64)  # 0表示无效
                # if len(tmp_agent_types) > 0:
                #     agent_types[:num_valid] = np.array(tmp_agent_types)[selected_indices]
                # frame_data['agent_types'] = agent_types  # 保存到帧数据
            
            frame_data['agent_status'] = agent_status       
            frame_data['agent_labels'] = agent_labels

            ### bev###
            bev_size = (128, 256)
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            patch_size = [102.4, 102.4]
            map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}
            point_cloud_range = np.array(point_cloud_range)

            # 替换原有gt_masks列表初始化
            semantic_mask = np.zeros((128, 256), dtype=np.uint8)  # 单通道语义图
            # gt_masks = []       # 存储BEV视角下的语义掩码
            # gt_labels = []      # 存储每个地图元素的类别标签
            # gt_bboxes = []      # 存储每个地图元素的边界框坐标

            town_name = frame_data['town_name']
            town_name = 'Town10' if town_name == 'Town10HD' else town_name  # 关键行：处理特殊城镇名
            
             # 提取该城镇的完整地图信息
            map_info = map_infos[town_name]
            lane_points = map_info['lane_points']                           # 车道线详细点集
            lane_sample_points = map_info['lane_sample_points']             # 车道线采样点
            lane_types = map_info['lane_types']                             # 车道类型标签
            trigger_volumes_points = map_info['trigger_volumes_points']     # 触发区域多边形点
            trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']   # 触发区域多边形采样点
            trigger_volumes_types = map_info['trigger_volumes_types']       # 触发区域类型

            # 获取世界坐标系到LiDAR坐标系的变换矩阵
            world2lidar = np.array(anno['sensors']['LIDAR_TOP']['world2lidar'])
            # 计算自车在LiDAR坐标系下的坐标 (x,y)
            ego_xy = np.linalg.inv(world2lidar)[0:2,3]
            
            #1st search # 第一阶段：处理车道线信息 -----------------------------------------------
            max_distance = 100          # 筛选车道线的最大距离阈值（单位：米）
            chosed_idx = []             # 存储符合距离要求的车道线索引

            # 遍历所有车道线采样点，筛选附近的车道
            for idx in range(len(lane_sample_points)):
                single_sample_points = lane_sample_points[idx]
                # 计算该车道所有采样点到自车的欧氏距离
                distance = np.linalg.norm((single_sample_points[:,0:2]-ego_xy),axis=-1)
                # 如果存在采样点距离自车小于阈值，则保留该车道
                if np.min(distance) < max_distance:
                    chosed_idx.append(idx)
            # 处理筛选后的车道线
            for idx in chosed_idx:
                # 过滤未定义类型的车道
                if not lane_types[idx] in map_element_class.keys():
                    continue

                # 坐标变换：将车道点从世界坐标系转到LiDAR坐标系
                points = lane_points[idx]
                points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
                points_in_ego = (world2lidar @ points.T).T
                #print(points_in_ego)
                # 筛选在感知范围内的点（根据点云范围参数）
                mask = (points_in_ego[:,0]>point_cloud_range[0]) & (points_in_ego[:,0]<point_cloud_range[3]) & (points_in_ego[:,1]>point_cloud_range[1]) & (points_in_ego[:,1]<point_cloud_range[4])
                points_in_ego_range = points_in_ego[mask,0:2]

                if len(points_in_ego_range) > 1:        # 需要至少两个点构成线
                    # 创建BEV掩码图像
                    # gt_mask = np.zeros(bev_size,dtype=np.uint8)
                    # 坐标归一化：将米制坐标转换为像素坐标
                    normalized_points = np.zeros_like(points_in_ego_range)
                    normalized_points[:,0] = (points_in_ego_range[:,0] + patch_size[0]/2)*(bev_size[0]/patch_size[0])
                    normalized_points[:,1] = (points_in_ego_range[:,1] + patch_size[1]/2)*(bev_size[1]/patch_size[1])

                    # 获取类别标签
                    gt_label =  map_element_class[lane_types[idx]]
                    # 在掩码上绘制车道线（折线）
                    cv2.polylines(semantic_mask, [normalized_points.astype(np.int32)], False, color=gt_label, thickness=2)

                    # gt_masks.append(gt_mask)
                    # gt_labels.append(gt_label)
                    # 计算边界框（基于掩码的非零区域）
                    # ys, xs = np.where(gt_mask==1)
                    # gt_bboxes.append([min(xs), min(ys), max(xs), max(ys)]) 
             # 第二阶段：处理触发区域（如交叉路口）-----------------------------------
            trigger_elements = []
            for idx in range(len(trigger_volumes_points)):
                # 过滤未定义类型的触发区域
                elem_type = trigger_volumes_types[idx]
                if elem_type not in map_element_class.keys():
                    continue
                trigger_elements.append((idx, elem_type))  # 收集有效元素
            
            # 定义优先级排序函数（数值越大越靠后处理）
            def trigger_priority(elem):
                _, elem_type = elem
                if elem_type == 'TrafficLight':
                    return 2   # 最高优先级
                elif elem_type == 'StopSign':
                    return 1    # 中优先级
                else:
                    return 0    # 低优先级
            # 按优先级升序排列（低->高），使高优先级元素最后处理
            trigger_elements_sorted = sorted(trigger_elements, key=trigger_priority)

            for elem in trigger_elements_sorted:
                idx, elem_type = elem
                # 坐标变换  将车道点从世界坐标系转到LiDAR坐标系
                points = trigger_volumes_points[idx]
                points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
                points_in_ego = (world2lidar @ points.T).T

                # 筛选在感知范围内的点
                mask = (points_in_ego[:,0]>point_cloud_range[0]) & (points_in_ego[:,0]<point_cloud_range[3]) & (points_in_ego[:,1]>point_cloud_range[1]) & (points_in_ego[:,1]<point_cloud_range[4])
                points_in_ego_range = points_in_ego[mask,0:2]

                if mask.all():      # 当所有点都在感知范围内时才处理
                    # 创建BEV掩码图像
                    # semantic_mask = np.zeros(bev_size,dtype=np.uint8)
                    # 坐标归一化：将米制坐标转换为像素坐标
                    normalized_points = np.zeros_like(points_in_ego_range)
                    normalized_points[:,0] = (points_in_ego_range[:,0] + patch_size[0]/2)*(bev_size[0]/patch_size[0])
                    normalized_points[:,1] = (points_in_ego_range[:,1] + patch_size[1]/2)*(bev_size[1]/patch_size[1])
                    # 获取类别标签
                    gt_label = map_element_class[elem_type]
                    # 填充多边形区域
                    cv2.fillConvexPoly(
                        semantic_mask, 
                        normalized_points.astype(np.int32), 
                        color=gt_label
                        )

                    # gt_masks.append(gt_mask)
                    # gt_labels.append(gt_label)
                    # 计算边界框
                    # ys, xs = np.where(gt_mask==1)
                    # gt_bboxes.append([min(xs), min(ys), max(xs), max(ys)]) 
            # 处理空结果的情况（添加占位数据）
            # if len(gt_masks) == 0:
            #     gt_masks.append(np.zeros(bev_size,dtype=np.uint8))
                # gt_labels.append(-1)    # 无效标签
                # gt_bboxes.append([0,0,0,0])
                
            # with open('/home/liwei/projects/Bench2DriveZoo/data/infos/b2d_map_infos.pkl', 'rb') as f: 
            #     map_infos = pickle.load(f)
            # frame_data['bev_semantic'] = generate_bev_semantic(frame_data, map_infos)

            if len(agent_status) == 0:
                continue
            
            # 生成BEV语义分割数据
            # town_name = folder_name.split('/')[1].split('_')[1]
            
            # last_position_dict = position_dict.copy()    
            # gt_ids = np.array(gt_ids)
            # gt_names = np.array(gt_names)
            # num_points_list = np.array(num_points_list)
            # gt_boxes = np.stack(gt_boxes)
            # npc2world = np.stack(npc2world_list)
            # frame_data['gt_ids'] = gt_ids
            # frame_data['gt_boxes'] = gt_boxes
            # frame_data['gt_names'] = gt_names
            # frame_data['num_points'] = num_points_list
            # frame_data['npc2world'] = npc2world
            # gt_masks = np.stack(gt_masks)
            frame_data['bev_semantic_map'] = semantic_mask
            final_data.append(frame_data)
    
    os.makedirs(join(OUT_DIR,tmp_dir),exist_ok=True)
    with open(join(OUT_DIR,tmp_dir,'b2d_infos_'+train_or_val+'_'+str(index)+'.pkl'),'wb') as f:
        pickle.dump(final_data,f)

    
        
def _xy_in_lidar(x, y):
    return (lidar_min_x <= x <= lidar_max_x) and (lidar_min_y <= y <= lidar_max_y)

def generate_infos(folder_list,workers,train_or_val,tmp_dir):

    folder_num = len(folder_list)
    devide_list = [(folder_num//workers)*i for i in range(workers)]
    devide_list.append(folder_num)
    for i in range(workers):
        sub_folder_list = folder_list[devide_list[i]:devide_list[i+1]]
        process = multiprocessing.Process(target=preprocess, args=(sub_folder_list,i,tmp_dir,train_or_val))
        process.start()
        process_list.append(process)
    for i in range(workers):
        process_list[i].join()
    union_data = []
    for i in range(workers):
        with open(join(OUT_DIR,tmp_dir,'b2d_infos_'+train_or_val+'_'+str(i)+'.pkl'),'rb') as f:
            data = pickle.load(f)
        union_data.extend(data)
    with open(join(OUT_DIR,'b2d_infos_'+train_or_val+'.pkl'),'wb') as f:
        pickle.dump(union_data,f)

if __name__ == "__main__":


    os.makedirs(OUT_DIR,exist_ok=True)
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--workers',type=int, default= 4, help='num of workers to prepare dataset')
    argparser.add_argument('--tmp_dir', default="tmp_data", )
    args = argparser.parse_args()    
    workers = args.workers
    process_list = []
    with open('/home/fqz02/Works/WangRunhu/demo02/navsim_workspace/dataset/Bench2Drive/bench2drive_base_train_val_split.json','r') as f:
        train_val_split = json.load(f)
        
    # all_folder = os.listdir(DATAROOT)
    all_folder = os.listdir(join(DATAROOT,'v1'))
    train_list = []
    val_list = []
    for foldername in all_folder:
        # if 'Town' in foldername and 'Route' in foldername and 'Weather' in foldername and not foldername in train_val_split['val']:
        if 'Town' in foldername and 'Route' in foldername and 'Weather' in foldername and not join('v1',foldername) in train_val_split['val']:
            # train_list.append(foldername)
            train_list.append(join('v1',foldername))
        else:
            # val_list.append(foldername)
            val_list.append(join('v1',foldername))

    print(val_list)
    print('processing train data...')
    generate_infos(train_list,workers,'train',args.tmp_dir)
    process_list = []
    print('processing val data...')
    generate_infos(val_list,workers,'val',args.tmp_dir)

    # print('processing map data...')
    # gengrate_map(MAP_ROOT)
    print('finish!')