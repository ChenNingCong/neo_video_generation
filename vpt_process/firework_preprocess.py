import torch
import numpy as np
import glob
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import namedtuple
from typing import Optional
@dataclass
class VideoDatasetInfo:
    is_latent : bool
    image_shape : Tuple
    vae_preprocessor : Any
    vae_postprocessor : Any
    vae : Any

from torch.utils.data import Dataset
class AdaptorDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        video = self.base_dataset[idx]["video"]
        return {"video": video, "labels": 0}

from functools import partial
@partial(jax.jit, backend="cuda", static_argnames="vae_model")
def decode(vae_model, vae_params, z):
    return vae_model.apply({"params":vae_params}, z, method=vae_model.decode)

DEFAULT_FILES = sorted(glob.glob((Path(__file__).parent / "./local_video/cheeky-cornflower-*.done").as_posix()))
# only get done file
DEFAULT_FILES = [i[:-len(".done")]+".vae" for i in DEFAULT_FILES]
def make_dataset_info(frame_rate: int = 8, files = DEFAULT_FILES, dtype=np.float32):
    def open_m(x):
        _m = np.memmap(x, mode="r", dtype=dtype).reshape(-1, 3, 5, 256)
        _m = _m.transpose(0, 3, 1, 2) # T, C, H, W
        return _m
    _ms = [open_m(i) for i in files]
    print(len(_ms))
    C, T, H, W = 256, frame_rate, 3, 5
    class VideoDataset(torch.utils.data.Dataset):
        def __init__(self):
            # remove T frames for each array
            self.l = torch.tensor([i.shape[0] - T for i in _ms])
            assert torch.all(self.l > 0)
            self.cum_l = self.l.cumsum(0)
        def __len__(self):
            return self.cum_l[-1].item()
        def __getitem__(self, i):
            # (T, C, H, W) to (C, T, H, W)
            video_id = torch.searchsorted(self.cum_l, torch.tensor(i), right=True)
            if video_id == 0:
                frame_id = i
            else:
                frame_id = i - self.cum_l[video_id-1]
            assert frame_id >= 0
            assert video_id < len(_ms)
            _m = _ms[video_id]
            return {"video":np.asarray(_m[frame_id:frame_id+T]).transpose(1, 0, 2, 3)}
        
    class JAXVAE:
        def __init__(self):
            self.vae_model = None
            self.vae_params = None
        def lazy(self):
            print("decoding!")
            from lucid_v1.models.vqvae import VQVAE
            import tensorflow as tf
            from lucid_v1.launch import hf_hub_download, vae_model_config
            from collections import namedtuple
            import ml_collections
            import pickle
            import os
            tmp_path = hf_hub_download(
                repo_id="ramimmo/lucidv1",
                filename="vae_params_numpy.tmp"
            )
            with tf.io.gfile.GFile(tmp_path, 'rb') as f:
                vae_params = pickle.loads(f.read())
            vae_model = VQVAE(ml_collections.FrozenConfigDict(vae_model_config), False)
            self.vae_model = vae_model 
            self.vae_params = vae_params
        def decode_frames(self, x):
            if self.vae_model is None:
                self.lazy()
            assert isinstance(x, torch.Tensor), f"Input must be a torch.Tensor, but got {type(x)}"
            assert (x.shape == (x.shape[0], 256, 3, 5)), f"Input tensor must have 4 dimensions (B C H W), but got {x.shape}"
            p = jnp.array(rearrange(x, 'b c h w->b h w c').cpu().numpy())
            frames = []
            for i in range(p.shape[0]):
                frames.append(decode(self.vae_model, self.vae_params, p[i:(i+1)])[0])
            p = jnp.stack(frames)
            p = np.asarray(p.astype(jnp.float32))
            p = torch.tensor(p, device=x.device).to(x.dtype)
            p = rearrange(p, 'b h w c->b c h w', b=x.shape[0])
            # torch.save(p, "test.pt")
            assert p.shape == (x.shape[0], 3, 384, 640)
            # create namedtuple for type compatibility
            # has a single field sample
            return p
        
        def decode(self, x):
            if self.vae_model is None:
                self.lazy()
            assert isinstance(x, torch.Tensor), f"Input must be a torch.Tensor, but got {type(x)}"
            assert (x.shape == (x.shape[0], 256, T, 3, 5)), f"Input tensor must have 5 dimensions (B C T H W), but got {x.shape}"
            p = jnp.array(rearrange(x, 'b c t h w->(b t) h w c').cpu().numpy())
            frames = []
            for i in range(p.shape[0]):
                frames.append(decode(self.vae_model, self.vae_params, p[i:(i+1)])[0])
            p = jnp.stack(frames)
            p = np.asarray(p.astype(jnp.float32))
            p = torch.tensor(p, device=x.device).to(x.dtype)
            p = rearrange(p, '(b t) h w c->b c t h w', b=x.shape[0], t =T)
            # torch.save(p, "test.pt")
            assert p.shape == (x.shape[0], 3, T, 384, 640)
            # create namedtuple for type compatibility
            # has a single field sample
            return namedtuple("VAEOutput", ["sample"])(p)
    dataset = AdaptorDataset(VideoDataset())
    vae = JAXVAE()

    dataset_info = VideoDatasetInfo(
        image_shape=(C, T, H, W), 
        is_latent=True,
        vae = vae,
        vae_preprocessor = lambda x : x,
        vae_postprocessor = namedtuple("PostProcessor", ["postprocess"])(lambda image, output_type : image))
    return dataset, dataset_info