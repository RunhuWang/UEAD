from PIL import Image
import numpy as np
import torch 
import torch.nn.functional
from torch.utils.data import Dataset
from torchvision import transforms as T

import pickle

class CARLA_Data(Dataset):

    def __init__(self, data_path):
        self.ego_x = []
        self.ego_y = []
        self.ego_yaw = []

        self.ego_vel = []       # 2
        self.ego_accel = []     # 2
        self.cmd_onehot = []    # 6

        self.img_paths = []
        self.lidar_paths = []
        self.waypoints = []

        self.agent_status = []
        self.agent_labels = []
        self.bev_semantic_map = []
    
        self._batch_read_number = 0

        # for sub_root in data_folders:
        print(f'load pkl data from {data_path}')
        data = self.load_pkl(data_path)

        # 将数据文件中的数据添加到相应的列表中
        for frame_idx in range(len(data)):
            frame_data = data[frame_idx]
            # self.ego_x.append(frame_data['ego_x'])
            # self.ego_y.append(frame_data['ego_y'])
            # self.ego_yaw.append(frame_data['ego_yaw'])
            self.ego_vel.append(frame_data['ego_vel'])
            self.ego_accel.append(frame_data['ego_accel'])
            self.cmd_onehot.append(frame_data['cmd_onehot'])
            self.img_paths.append(frame_data['sensors']['cam_data_path'])
            self.lidar_paths.append(frame_data['sensors']['lidar_2d_npy_path'])
            self.waypoints.append(frame_data['waypoints'])
            self.agent_status.append(frame_data['agent_status'])
            self.agent_labels.append(frame_data['agent_labels'])
            # self.frame_idx.append(frame_data['frame_idx'])
            self.bev_semantic_map.append(frame_data['bev_semantic_map'])

        # 转换为 NumPy 数组 (假设所有元素形状相同)
        # self.ego_x = np.array(self.ego_x, dtype=np.float32)
        # self.ego_y = np.array(self.ego_y, dtype=np.float32)
        # self.ego_yaw = np.array(self.ego_yaw, dtype=np.float32)
        self.cmd_onehot = np.array(self.cmd_onehot, dtype=np.float32)
        self.ego_vel = np.array(self.ego_vel, dtype=np.float32)
        self.ego_accel = np.array(self.ego_accel, dtype=np.float32)
        self.waypoints = np.array(self.waypoints, dtype=np.float32)
        self.agent_status = np.array(self.agent_status, dtype=np.float32)
        self.agent_labels = np.array(self.agent_labels, dtype=bool)
        self.bev_semantic_map = np.array(self.bev_semantic_map, dtype=np.float32)


        # 定义图像预处理
        self.im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    
    def load_pkl(self,file_path):
        with open(file_path, 'rb') as f:
            dict = pickle.load(f)
        return dict

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.ego_vel)
    
    def __getitem__(self, idx):
        """Returns the item at index idx. """
        # 基础数据加载
        # ego_x = torch.tensor(self.ego_x[idx], dtype=torch.float)        # 解决标量问题
        # ego_y = torch.tensor(self.ego_y[idx], dtype=torch.float)        # 解决标量问题
        # ego_yaw = torch.tensor(self.ego_yaw[idx], dtype=torch.float)    # 解决标量问题
        # frame_idx = torch.tensor(self.frame_idx[idx], dtype=torch.float)    # 解决标量问题
        cmd_onehot = torch.torch.as_tensor(self.cmd_onehot[idx])            # 确保cmd_onehot是数组
        ego_vel = torch.torch.as_tensor(self.ego_vel[idx])                   # 确保ego_vel是数组
        ego_accel = torch.torch.as_tensor(self.ego_accel[idx])              # 确保ego_accel是数组
        waypoints = torch.torch.as_tensor(self.waypoints[idx])              # 确保waypoints是数组
        agent_status = torch.torch.as_tensor(self.agent_status[idx])        # 确保agent_status是数组
        agent_labels = torch.torch.as_tensor(self.agent_labels[idx])          # 确保agent_labels是数组
        bev_semantic_map = torch.torch.as_tensor(self.bev_semantic_map[idx])

        img = Image.open(self.img_paths[idx])
        img = self.im_transform(img)

        lidar = torch.torch.as_tensor(np.load(self.lidar_paths[idx]))

         # ==================== 结构化分组 ====================
         # 第一部分：特征集合
        features_dict  = {
                "camera_feature": img,  # [C, H, W]
                "lidar_feature": lidar,  # [H, W] 或其他维度
                "status_feature": torch.cat([
                    cmd_onehot,     # [6]
                    ego_vel,        # [2]
                    ego_accel       # [2]
                ], dim=0)           # -> [10]
            }
            
            # 第二部分：目标集合
        targets_dict = {
                "trajectory": waypoints,    # [8, 2]
                "agent_states": agent_status,  #  torch.Size([64, max_agents, 5])而不是torch.Size([32, 1, max_agents, 5])
                "agent_labels": agent_labels,   # torch.Size([64, max_agents])而不是torch.Size([32, 1, max_agents])
                "bev_semantic_map":bev_semantic_map
            }

        return (features_dict,targets_dict)
