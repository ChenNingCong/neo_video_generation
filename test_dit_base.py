# %load_ext autoreload
# %autoreload 3
import torch
import os
import tiny_trainer
from tiny_trainer import *
from video_model import DiTModelWrapper
from videogpt.data import preprocess
from dataclasses import dataclass
import wandb
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import os
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
from videogpt.data import preprocess
import torch
from typing import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from streaming.vision.base import StreamingDataset
from videogpt.vqvae import VQVAE
from torch.utils.data.distributed import DistributedSampler


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


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_l):
        self.dataset = dataset
        self.base_l = base_l
        self.l = len(self.dataset)
    def __len__(self):
        return self.l
    def __getitem__(self, i):
        return self.dataset[i %self.base_l]



@dataclass
class VideoDatasetInfo:
    is_latent : bool
    image_shape : Tuple
class MNISTFactory(AbstractTrainerFactory):
    def __init__(self, world_size, per_device_batch_size, use_single=False):
        self.per_device_batch_size = per_device_batch_size
        self.world_size = world_size
        self.use_single = use_single
    def make_model(self):
        model = DiTModelWrapper(
            num_attention_heads = 4,
            attention_head_dim = 64,
            in_channels = 1,
            out_channels = 1,
            num_layers = 8,
            dropout = 0.0,
            norm_num_groups = 16,
            attention_bias = True,
            spatial_size = 64,
            temporal_size = 16,
            spatial_patch_size = 4,
            temporal_patch_size = 4,
            num_embeds_ada_norm = None,
            class_condition=False
        )
        return model
    def make_optimizer(self, model):
        return ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr = 3e-4, weight_decay=0.0)
    def make_dataloader(self, rank : int): 
        per_device_batch_size = self.per_device_batch_size
        dataset = MovingMNISTGrayDataset(data_dir = "../datasets/", sequence_length=16, train=True, resolution=64)
        if self.use_single:
            dataset = SingleDataset(dataset, base_l=64)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, drop_last=True)
        if self.world_size == 1:
            sampler = None
        else:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True, seed = 0, drop_last=True)
        return VideoDatasetInfo(image_shape=(1, 16, 64, 64), is_latent=False), dataloader, sampler
    def make_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=500) 

from tiny_trainer import register_config, MISSING
from dataclasses import dataclass
@register_config(group="diffusion", name="base")
@dataclass
class DiffusionConfig:
    sample_type : str = "lognorm0_1"
    max_timestep : int = 512
    sampling_step : int = 50

@torch.no_grad
def sample_image(rank : int, model, max_timestep: int, sampling_step : int, batch_size, class_labels, datasetinfo, use_classlabel, device):
    image_shape = datasetinfo.image_shape
    # we add a generator to better monitor the quality here
    X0 = torch.randn((batch_size, *image_shape)).to(device)
    for i in range(sampling_step):
        val = (i / sampling_step) * max_timestep
        # we use a uniform sampling here
        timestep = torch.full(size=(batch_size,), fill_value=val).to(device)
        if use_classlabel:
            pred = model(X0, timestep=timestep, class_labels = class_labels)
        else:
            pred = model(X0, timestep=timestep)
        X0 += pred.sample * (1 / sampling_step)
    if datasetinfo.is_latent:
        X0 = datasetinfo.vae_preprocessor(X0)
        X0 = datasetinfo.vae.decode(X0).sample
    return X0


@torch.no_grad
def display_video_tensor(video, fps=4):
    # Ensure tensor is on CPU
    # B, C, T, H, W
    # 4, 3, 16, 64, 64
    # the input should be in [0, 1] !
    if video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1, 1)
    video = torch.clip(video, 0, 1).transpose(1,2)
    # Ensure the video is in uint8 format
    assert video.shape == (4, 16, 3, 64, 64)
    video = (video * 255).to(torch.uint8)
    video = video.cpu().detach().numpy()
    # Display the video
    video = wandb.Video(video, fps=fps)
    wandb.log({"video": video}, commit=False)

class MNISTTrainer(DefaultTrainer):
    def __init__(self, diff_config : DiffusionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diff_config = diff_config
        self.loss_fun = torch.nn.MSELoss()
    def eval_model(self,i, is_debug = False):
        model = self.model
        model.eval()
        def normalize_image(x):
            return x + 0.5
        datasetinfo = self.datasetinfo
        world_size = self.world_size
        max_timestep = self.diff_config.max_timestep
        rank = self.rank
        sampling_step = self.diff_config.sampling_step
        use_classlabel = True
        device = rank
        total_batch_size = 4
        class_labels = None
        step = total_batch_size // world_size
        batch_size = step
        image_array = sample_image(rank,
                                   model, 
                                   max_timestep,
                                   sampling_step,
                                   batch_size = batch_size, 
                                   class_labels = class_labels,
                                   datasetinfo = datasetinfo,
                                   use_classlabel = use_classlabel,
                                   device= device)
        
        image_arrays = [torch.zeros_like(image_array, device=rank) for _ in range(world_size)]
        dist.all_gather(image_arrays, image_array)
        if rank == 0:
            image_array = torch.cat(image_arrays, dim = 0)
            if datasetinfo.is_latent:
                image_array = datasetinfo.vae_postprocessor.postprocess(image = image_array, output_type='pt')
            else:
                image_array = normalize_image(image_array)
            display_video_tensor(image_array)
    def prepare_input(self, data):
        image = data["video"].to(self.rank)
        device = self.rank
        # (B, C, T, H, W)
        noise = torch.randn(image.shape).to(device)
        sample_type = self.diff_config.sample_type
        max_timestep = self.diff_config.max_timestep
        if sample_type == "uniform":
            timestep = torch.rand(size=(image.size(0),)).to(device) * max_timestep
        elif sample_type == "lm":
            x = torch.tensor([lmpdf((i + 1/2)/max_timestep) + 0.1 for i in range(max_timestep)]).to(device)
            x = x/x.sum()
            timestep = torch.multinomial(x, image.size(0)).to(torch.float32) # [0, max_timestep-1]
            timestep += torch.rand(size=(image.size(0),)).to(device) # [0, max_timestep)
        elif sample_type == "lognorm0_1":
            x = torch.randn(size=(image.size(0),)).to(device)
            # map [-inf, inf] to [0, 1]
            x = torch.sigmoid(x)
            # scale the timestep into [0, max_timestep]
            timestep = x * max_timestep
        alpha = (timestep / max_timestep).view(-1, *([1]*(len(image.shape) - 1)))
        point = (1 - alpha) * noise + alpha * image
        target = image - noise
        return {"point" : point, "target" : target, "timestep" : timestep}
    def calculate_loss(self, point, timestep, target):
        model = self.model
        output = model(point, timestep=timestep, class_labels = None)
        sample = output.sample
        return self.loss_fun(sample, target)

if __name__ == "__main__":
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="mnist_config_base")
        trainer_cfg = cfg.trainer
        factory = MNISTFactory(world_size = trainer_cfg.world_size, 
                               per_device_batch_size = 32 // trainer_cfg.world_size,
                               use_single=False)
        trainer = MNISTTrainer(cfg.diffusion, factory, trainer_cfg)
        trainer.run()