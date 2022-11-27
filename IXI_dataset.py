import random
import pathlib
import scipy.io as sio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import normalize_zero_to_one
 
from torch.utils import data as Data 
from dataprocess import complex2pseudo,kspace2image
'''
input: data_path, u_mask_path, s_mask_up_path, s_mask_down_path, sample_rate
output:label, self.mask_under, self.mask_net_up, self.mask_net_down,    (BCHW)

'''



def arbitrary_dataset_split(dataset ,
                            indices_list 
                            ) :
    return [Data.Subset(dataset, indices) for indices in indices_list]


def datasets2loaders(datasets   ,
                     *,
                     batch_size  = (1, 1, 1),  # train, val, test
                     is_shuffle  = (True, False, False),  # train, val, test  should be change (True, False, False)
                     num_workers  = 0)  :
    """
    a tool for build N-datasets into N-loaders
    """
   
    assert isinstance(datasets[0], Data.Dataset)
    n_loaders = len(datasets)
    assert n_loaders == len(batch_size)
    assert n_loaders == len(is_shuffle)

    loaders = []
    for i in range(n_loaders):
        loaders.append(
            Data.DataLoader(datasets[i], batch_size=batch_size[i], shuffle=is_shuffle[i], num_workers=num_workers)
        )

    return loaders

'''
input dataset
output train val test loader
'''
def build_loader(dataset, batch_size,
                 train_indices=np.arange(0,300),
                 val_indices=np.arange(300,330),
                 test_indices=np.arange(330, 360),
                 num_workers=4):
    """
    :return: train/validation/test loader
    """
    print('dataset.__len__:',dataset.__len__)
    datasets = arbitrary_dataset_split(dataset, [train_indices, val_indices, test_indices])
    loaders = datasets2loaders(datasets, batch_size=(batch_size,) * 3, is_shuffle=(True, False, False), #should be change is_shuffle=(True, False, False)
                               num_workers=num_workers)
    return loaders



'''
input path of 数据 模拟欠采mask 上下划分mask
数据采样比例 选取中间多少slice进行重建 sample_rate
'''

class IXIData(Dataset):
    def __init__(self, data_path, u_mask_path, s_mask_up_path, s_mask_down_path):
        super(IXIData, self).__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        # self.sample_rate = sample_rate  用于取每个病人的中间部分

        self.examples = []
        # files = list(pathlib.Path(self.data_path).iterdir())
        # The middle slices have more detailed information, so it is more difficult to reconstruct. 选取中间部分进行重建？
        # start_id, end_id = 30, 100
        # for file in sorted(files):
        #     self.examples += [(file, slice_id) for slice_id in range(start_id, end_id)]
        # if self.sample_rate < 1:
        #     random.shuffle(self.examples)
        #     num_examples = round(len(self.examples) * self.sample_rate)
        #     self.examples = self.examples[:num_examples]
        
        data_dict = np.load(data_path)
        # loading dataset
        kspace = data_dict['kspace']  # List[ndarray]
        self.images=kspace2image(kspace)
        self.images=complex2pseudo(self.images)
        self.examples = self.images.astype(np.float32)


        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # file, slice_id = self.examples[item]
        # data = nib.load(file)
        label = self.examples[item]
        label = normalize_zero_to_one(label, eps=1e-6)  #归一化部分 是否需要额外进行归一化？ 
        label = torch.from_numpy(label)
       
       
        # self.mask_under, self.mask_net_up, self.mask_net_down=self.mask_under.permute(2,0,1), self.mask_net_up.permute(2,0,1), self.mask_net_down.permute(2,0,1)
    
        
        return label, self.mask_under, self.mask_net_up, self.mask_net_down #, file.name, slice_id
