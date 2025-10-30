# %load_ext autoreload
# %autoreload 3
import torch
from video_model_rope_cond_general import DiTModelWrapper
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
from data_script.wand_display_video import display_mnist_video_tensor
from dataclasses import dataclass
BATCH_SIZE = 8
"""Moving MNIST dataset from http://www.cs.toronto.edu/~nitish/unsupervised_video.

Augments the original Moving MNIST dataset with labels for text guided video diffusion.

Based on an implementation of unconditional MovingMNIST generation from:
https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe
"""

from bs4 import BeautifulSoup
import numpy as np
import os
import requests
import sys
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Callable, List, Tuple


def load_moving_mnist_image(
    training_height: int,
    training_width: int,
    split: str = "train",
    invert: bool = False,
) -> Tuple[Dataset, Callable[[torch.Tensor], List[str]]]:
    assert split in ["train", "validation"]

    if invert:
        xforms = [
            # To the memory requirements, resize the MNIST
            # images from (64,64) to (32, 32).
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Convert the motion images to (0,1) float range
            v2.ToDtype(torch.float32, scale=True),
            # Invert the dataset for LoRA training
            v2.Lambda(_invert),
        ]
    else:
        xforms = [  # To the memory requirements, resize the MNIST
            # images from (64,64) to (32, 32).
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Convert the motion images to (0,1) float range
            v2.ToDtype(torch.float32, scale=True),
        ]
    if split == "train":
        dataset = MovingMNISTImage(
            ".",
            train=True,
            transform=v2.Compose(xforms),
        )

    else:
        dataset = MovingMNISTImage(
            ".",
            train=False,
            transform=v2.Compose(xforms),
        )
    return dataset, convert_labels_to_prompts


def load_moving_mnist(
    training_height: int,
    training_width: int,
    split: str = "train",
    invert: bool = False,
) -> Tuple[Dataset, Callable[[torch.Tensor], List[str]]]:
    assert split in ["train"]

    if invert:
        xforms = [
            # To the memory requirements, resize the MNIST
            # images from (64,64) to (32, 32).
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Convert the motion images to (0,1) float range
            v2.ToDtype(torch.float32, scale=True),
            # Invert the dataset for LoRA training
            v2.Lambda(_invert),
        ]
    else:
        xforms = [  # To the memory requirements, resize the MNIST
            # images from (64,64) to (32, 32).
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Convert the motion images to (0,1) float range
            v2.ToDtype(torch.float32, scale=True),
        ]

    dataset = MovingMNIST(
        ".",
        transform=v2.Compose(xforms),
    )
    return dataset, convert_labels_to_prompts


class MovingMNIST(Dataset):
    """Moving MNIST dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Download the data to the root dir if it does not exist
        from urllib.request import urlretrieve

        def download(filename, source_url, file_id):
            print(f"Downloading {source_url} to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            download_file_from_google_drive(file_id, filename)

        videos_file_name = os.path.join(root_dir, "MovingMNIST/videos_data.npz")
        labels_file_name = os.path.join(root_dir, "MovingMNIST/labels_data.npz")

        _VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"
        _VIDEOS_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _LABELS_FILE_ID = "17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _VIDEOS_URL, _VIDEOS_FILE_ID)
        if not os.path.isfile(labels_file_name):
            download(labels_file_name, _LABELS_URL, _LABELS_FILE_ID)

        self._num_frames_per_video = 30
        self._num_videos = 10000
        self._num_digits_per_video = 2

        with np.load(videos_file_name, allow_pickle=True) as npz:
            videos_np = npz[npz.files[0]]

        with np.load(labels_file_name, allow_pickle=True) as npz:
            labels_np = npz[npz.files[0]]

        self._video_data = torch.from_numpy(
            videos_np.reshape(self._num_videos, 1, self._num_frames_per_video, 64, 64)
        )
        self._labels_data = torch.from_numpy(
            labels_np.reshape(
                self._num_videos, self._num_frames_per_video, self._num_digits_per_video
            )[:, 0, :]
        ).squeeze()

    def __len__(self):
        return self._video_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self._video_data[idx]
        labels = self._labels_data[idx]
        if self.transform:
            video = self.transform(video)
        return video, labels


class MovingMNISTImage(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, train: bool = True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Download the data to the root dir if it does not exist
        from urllib.request import urlretrieve

        def download(filename, source_url, file_id):
            print(f"Downloading {source_url} to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            download_file_from_google_drive(file_id, filename)

        train_videos_file_name = os.path.join(root_dir, "MovingMNIST/videos_data.npz")
        train_labels_file_name = os.path.join(root_dir, "MovingMNIST/labels_data.npz")
        val_videos_file_name = os.path.join(
            root_dir, "MovingMNIST/videos_data_validation.npz"
        )
        val_labels_file_name = os.path.join(
            root_dir, "MovingMNIST/labels_data_validation.npz"
        )

        _TRAIN_VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq&confirm=1"
        _TRAIN_LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai&confirm=1"
        _TRAIN_VIDEOS_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _TRAIN_LABELS_FILE_ID = "17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        _VAL_VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq&confirm=1"
        _VAL_LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai&confirm=1"
        _VAL_VIDEOS_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _VAL_LABELS_FILE_ID = "17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        if train:
            videos_file_name = train_videos_file_name
            labels_file_name = train_labels_file_name

            _VIDEOS_URL = _TRAIN_VIDEOS_URL
            _LABELS_URL = _TRAIN_LABELS_URL
            _VIDEOS_FILE_ID = _TRAIN_VIDEOS_FILE_ID
            _LABELS_FILE_ID = _TRAIN_LABELS_FILE_ID
        else:
            videos_file_name = val_videos_file_name
            labels_file_name = val_labels_file_name
            _VIDEOS_URL = _VAL_VIDEOS_URL
            _LABELS_URL = _VAL_LABELS_URL
            _VIDEOS_FILE_ID = _VAL_VIDEOS_FILE_ID
            _LABELS_FILE_ID = _VAL_LABELS_FILE_ID

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _VIDEOS_URL, _VIDEOS_FILE_ID)
        if not os.path.isfile(labels_file_name):
            download(labels_file_name, _LABELS_URL, _LABELS_FILE_ID)

        self._num_frames_per_video = 30
        self._num_videos = 10000
        self._num_digits_per_video = 2

        with np.load(videos_file_name) as npz:
            videos_np = npz[npz.files[0]]

        with np.load(labels_file_name) as npz:
            labels_np = npz[npz.files[0]]

        # The video data is (num_videos * num_frames, 1, 64, 64)
        self._video_data = torch.from_numpy(videos_np)
        self._labels_data = torch.from_numpy(labels_np).squeeze()

    def __len__(self):
        return self._video_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self._video_data[idx]
        labels = self._labels_data[idx]
        if self.transform:
            video = self.transform(video)
        return video, labels


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_length = int(response.headers["Content-Length"])
        total_chunks = (
            (content_length // CHUNK_SIZE) + 1
            if content_length % CHUNK_SIZE != 0
            else 0
        )
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total_chunks):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    content_type = response.headers["Content-Type"]
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    else:
        if content_type.startswith("text/html"):
            # Second download for large file virus warning
            html_content = response.text
            assert html_content.startswith(
                "<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning"
            )

            soup = BeautifulSoup(html_content, features="html.parser")
            form_tag = soup.find("form", {"id": "download-form"})
            download_url = form_tag["action"]

            # Get all of the attributes
            id = soup.find("input", {"name": "id"})["value"]
            export = soup.find("input", {"name": "export"})["value"]
            confirm = soup.find("input", {"name": "confirm"})["value"]
            uuid = soup.find("input", {"name": "uuid"})["value"]
            params = {
                "id": id,
                "export": export,
                "confirm": confirm,
                "uuid": uuid,
            }
            response = session.get(download_url, params=params, stream=True)
    save_response_content(response, destination)


def convert_labels_to_prompts(labels: torch.Tensor) -> List[str]:
    """Converts MNIST class labels to text prompts.

    Supports both the strings "0" and "zero" to describe the
    class labels.
    """
    # The conditioning we pass to the model will be a vectorized-form of
    # MNIST classes. Since we have a fixed number of classes, we can create
    # a hard-coded "embedding" of the MNIST class label.
    text_labels = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ]

    # First convert the labels into a list of string prompts
    prompts = [
        f"{text_labels[labels[i][0]][torch.randint(0, len(text_labels[labels[i][0]]), size=())]} and {text_labels[labels[i][1]][torch.randint(0, len(text_labels[labels[i][1]]), size=())]}"
        for i in range(labels.shape[0])
    ]
    return prompts


def _invert(x: torch.Tensor) -> torch.Tensor:
    return v2.functional.invert(x)

import os
import contextlib


@contextlib.contextmanager
def new_cd(x):
    d = os.getcwd()

    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)
class OVerlayDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        video, labels = self.base_dataset[idx]
        video = video.squeeze(0)  # (T, H, W)
        start = np.random.randint(low=0, high=len(video) - 16)
        
        # take a slice of the video
        video = video[start:start + 16]
        # pair encoding
        labels = 10 * labels[0] + labels[1]
        return {"video":(video).unsqueeze(0), "labels":labels}
class MNISTFactory(AbstractTrainerFactory):
    def __init__(self, world_size, per_device_batch_size, use_single=False):
        self.per_device_batch_size = per_device_batch_size
        self.world_size = world_size
        self.use_single = use_single
    def make_model(self):
        model = DiTModelWrapper(
            num_attention_heads = 16,
            attention_head_dim = 32,
            in_channels = 256,
            out_channels = 256,
            num_layers = 12,
            dropout = 0.0,
            norm_num_groups = 16, # not used
            attention_bias = True,
            spatial_size = (3, 5),
            temporal_size = 16,
            spatial_patch_size = (1, 1),
            temporal_patch_size = 1,
            num_embeds_ada_norm = None,
            class_condition=False,
            num_classes=0
        )
        return model
    def make_optimizer(self, model : DiTTransformer):

        return ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr = 2e-4, weight_decay=0.0)
    def make_dataloader(self, rank : int): 
        per_device_batch_size = self.per_device_batch_size
        from vpt_process.firework_preprocess import make_dataset_info
        dataset, dataset_info = make_dataset_info(16)
        if self.use_single:
            dataset = SingleDataset(dataset, base_l=64)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, drop_last=True, num_workers=4)
        if self.world_size == 1:
            sampler = None
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, drop_last=True, num_workers=4, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, drop_last=True, num_workers=4, shuffle=False)
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True, seed = 0, drop_last=True)
        return dataset_info, dataloader, sampler
    def make_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        WARMUP_STEPS = 500
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1,              # End factor is 1.0 to reach BASE_LR
            total_iters=WARMUP_STEPS    # The total number of steps/epochs for warmup
        )
        return warmup_scheduler

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
        pred = model(X0, timestep=timestep, class_labels = class_labels)
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
            return x
        datasetinfo = self.datasetinfo
        world_size = self.world_size
        max_timestep = self.diff_config.max_timestep
        rank = self.rank
        sampling_step = self.diff_config.sampling_step
        use_classlabel = False
        device = rank
        total_batch_size = 4
        step = total_batch_size // world_size
        batch_size = step
        class_labels = None
        image_array = sample_image(rank,
                                   model, 
                                   max_timestep,
                                   sampling_step,
                                   batch_size = batch_size, 
                                   class_labels = class_labels,
                                   datasetinfo = datasetinfo,
                                   use_classlabel = use_classlabel,
                                   device= device)
        if rank == 0:
            image_array = image_array
            if datasetinfo.is_latent:
                image_array = datasetinfo.vae_postprocessor.postprocess(image = image_array, output_type='pt')
            else:
                image_array = normalize_image(image_array)
            # image_array is of shape B, C, T, H, Wimage_array = torch.clamp(image_array, 0, 1) * 255
            import math
            image_array = torch.clamp(image_array, 0, 1) * 255
            from einops import rearrange
            from torchvision.io import write_video
            total_batch_size = image_array.shape[0]
            rows = int(math.sqrt(total_batch_size))
            cols =  total_batch_size // rows
            image_array = rearrange(
                    image_array,
                    '(x y) c t h w -> t (x h) (y w) c',
                    x=rows, y=cols
                )
            # torch.save(results, "help.pth")
            write_video("test.mp4", image_array.cpu().detach(), fps=8, options={'crf': '10'})
            video = wandb.Video(data_or_path="test.mp4")
            wandb.log({"video": video}, commit=False)
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
        # class_labels = data["labels"].to(self.rank).to(torch.long)
        return {"point" : point, "target" : target, "timestep" : timestep, "class_labels" : None}
    def calculate_loss(self, point, timestep, target, class_labels):
        model = self.model
        output = model(point, timestep=timestep, class_labels = None)
        sample = output.sample
        return self.loss_fun(sample, target)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    with initialize(version_base=None, config_path="config/mnist"):
        cfg = compose(config_name=args.config)
        trainer_cfg = cfg.trainer
        factory = MNISTFactory(world_size = trainer_cfg.world_size, 
                               per_device_batch_size = BATCH_SIZE // trainer_cfg.world_size,
                               use_single=cfg.use_single)
        trainer = MNISTTrainer(cfg.diffusion, factory, trainer_cfg)
        trainer.run()