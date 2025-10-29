import math
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset 

def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video

class MovingMNISTGrayDataset(Dataset):
    def __init__(self, data_dir, sequence_length, train=True, resolution=64):
        super().__init__()
        # moving mnist split the dataset by frames, not videos, this is counter-intuitive
        
        self.dataset = torchvision.datasets.MovingMNIST(root=data_dir, download=True)
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        l = int(0.9 * len(self.dataset))
        if self.train:
            self.offset = 0
            self.l = l
        else:
            self.offset = l
            self.l = len(self.dataset) - l

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for MovingMNISTDataset dataset')

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        assert idx < self.l
        start = np.random.randint(low=0, high=20 - self.sequence_length)
        # T, C, H, W
        video = self.dataset[idx + self.offset][start:start + self.sequence_length].permute((0, 2,3,1))
        assert video.shape[0] == self.sequence_length
        # preprocess accepts only T, H, W, C inputs
        # we repeat the input 3 times...it seems that this requires fewer modification
        #video = video.repeat(1,1,1,3)
        return dict(video=preprocess(video, self.resolution))
