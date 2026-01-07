import numpy as np
from torch.utils.data import Dataset
from feeders import tools

# Định nghĩa cặp xương NTU (0-based index) để không phụ thuộc file ngoài
ntu_pairs = (
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
    (23, 24), (24, 11)
)

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Giữ nguyên logic load data của bạn
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        # Giữ nguyên
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # Loop fix data lỗi (Giữ nguyên logic của bạn)
        while valid_frame_num == 0:
            index = (index + 1) % len(self.data)
            data_numpy = self.data[index]
            label = self.label[index]
            data_numpy = np.array(data_numpy)
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # Pre-processing (Crop/Resize)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # Random Rotation augmentation
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # ===============================================================
        # SỬA ĐỔI QUAN TRỌNG: 9-CHANNEL FUSION
        # ===============================================================
        
        # 1. Luôn giữ Joint data gốc (3 kênh)
        joint_data = data_numpy 
        
        channels = [joint_data] # Bắt đầu danh sách kênh với Joint

        # 2. Tính Bone (Nếu bật) -> Thêm 3 kênh
        if self.bone:
            bone_data = np.zeros_like(joint_data)
            for v1, v2 in ntu_pairs:
                # v1, v2 là index chuẩn (0-based)
                bone_data[:, :, v1, :] = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]
            channels.append(bone_data)

        # 3. Tính Velocity (Nếu bật) -> Thêm 3 kênh
        if self.vel:
            vel_data = np.zeros_like(joint_data)
            vel_data[:, :-1] = joint_data[:, 1:] - joint_data[:, :-1]
            vel_data[:, -1] = 0
            channels.append(vel_data)

        # 4. GỘP TẤT CẢ (Concatenate)
        # Nếu bật cả bone và vel: 3 + 3 + 3 = 9 kênh
        data_numpy = np.concatenate(channels, axis=0)
        
        # ===============================================================

        return data_numpy, label, index

    def top_k(self, score, top_k):
        # Giữ nguyên
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod