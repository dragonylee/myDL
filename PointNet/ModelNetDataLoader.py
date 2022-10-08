import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def pc_normalize(pc):
    """
    点云标准化（重心在原点，点云在单位球内部）
    :param pc:
    :return:
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, n_point):
    """
    最远点采样：
    首先随机选择一个点加入S，然后不断重复，每次选择 distance to S 最远的顶点加入S，
    直到size(S)达到目标，得到的S即为需要的采样点集。
    :param point:
    :param n_point:
    :return:
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((n_point,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 n_points=1024,
                 process_data=False,
                 use_uniform_sample=False,
                 use_normals=False,
                 num_category=40):
        """
        目前文件中的数据集已经resample过了，直接用即可。
        如果需要修改，例如变化顶点数目，可以视情况使用process_data和use_uniform_sample参数。

        :param split: 切分（'train'或'test'）
        :param root: 数据集路径
        :param n_points: 点云顶点数目
        :param process_data: 处理、（第一次）离线存储并使用数据
        :param use_uniform_sample: 使用均匀采样
        :param use_normals: 使用法向量
        :param num_category: 类别数目（10或40）
        """
        self.root = root
        self.n_points = n_points
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.n_points))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.n_points))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.n_points)
                    else:
                        point_set = point_set[0:self.n_points, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.n_points)
            else:
                point_set = point_set[0:self.n_points, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == "__main__":
    import torch

    data = ModelNetDataLoader('./../data/modelnet40_normal_resampled/', split='train')

    points, labels = data.__getitem__(0)
    print(points.shape)
