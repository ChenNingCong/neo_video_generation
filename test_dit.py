# %load_ext autoreload
# %autoreload 3
import torch
from video_model import DiTModelWrapper
from dataclasses import dataclass
import wandb
import torch.distributed as dist
from hydra import compose, initialize
import torch
import torch
from typing import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from tiny_trainer import register_config, MISSING, AbstractTrainerFactory, DefaultTrainer
from torch.utils.data.distributed import DistributedSampler
from data_script.mnist import MovingMNISTGrayDataset
from data_script.single_dataset import SingleDataset
from data_script.dataset_info import VideoDatasetInfo
from data_script.wand_display_video import display_video_tensor
from dataclasses import dataclass

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
            norm_num_groups = 16, # not used
            attention_bias = True,
            spatial_size = 32,
            temporal_size = 16,
            spatial_patch_size = 4,
            temporal_patch_size = 2,
            num_embeds_ada_norm = None,
            class_condition=False
        )
        return model
    def make_optimizer(self, model):
        return ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr = 1e-4, weight_decay=0.0)
    def make_dataloader(self, rank : int): 
        per_device_batch_size = self.per_device_batch_size
        dataset = MovingMNISTGrayDataset(data_dir = "./datasets/", sequence_length=16, train=True, resolution=32)
        if self.use_single:
            dataset = SingleDataset(dataset, base_l=64)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, drop_last=True)
        if self.world_size == 1:
            sampler = None
        else:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True, seed = 0, drop_last=True)
        return VideoDatasetInfo(image_shape=(1, 16, 32, 32), is_latent=False), dataloader, sampler
    def make_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=500) 

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
        cfg = compose(config_name="mnist_config_64")
        trainer_cfg = cfg.trainer
        factory = MNISTFactory(world_size = trainer_cfg.world_size, 
                               per_device_batch_size = 32 // trainer_cfg.world_size,
                               use_single=True)
        trainer = MNISTTrainer(cfg.diffusion, factory, trainer_cfg)
        trainer.run()