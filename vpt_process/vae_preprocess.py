print("decoding!")
from lucid_v1.models.vqvae import VQVAE
import tensorflow as tf
from lucid_v1.launch import hf_hub_download, vae_model_config
from collections import namedtuple
import ml_collections
import pickle
import os
import jax
tmp_path = hf_hub_download(
    repo_id="ramimmo/lucidv1",
    filename="vae_params_numpy.tmp"
)
with tf.io.gfile.GFile(tmp_path, 'rb') as f:
    vae_params = pickle.loads(f.read())
vae_model = VQVAE(ml_collections.FrozenConfigDict(vae_model_config), False)
from functools import partial
@partial(jax.jit, backend="cuda", static_argnames="vae_model")
def decode(vae_model, vae_params, z):
    return vae_model.apply({"params":vae_params}, z, method=vae_model.decode)
@partial(jax.jit, backend="cuda")
def encode_video(vae_params, video):
    quantized, result_dict = vae_model.apply({"params":vae_params}, video, method=vae_model.encode)
    return quantized, result_dict
from typing import List
import numpy as np
import jax.numpy as jnp
import math
from tqdm.autonotebook import tqdm
def encode_one_video(video, vae_video_path : str, frame_batch_size = 16):
    total_frames = int(video.fps * video.duration)
    total_batch = math.floor(total_frames / frame_batch_size)
    remained_num_frame = total_frames - (frame_batch_size*total_batch)
    num_frame_processed = 0
    num_batch_processed = 0
    # we use bfloat16 as storage and jax computation format
    # if os.path.exists(vae_video_path):
    #     os.remove(vae_video_path)
    vae_video_np = np.memmap(str(vae_video_path), dtype=np.float32, mode='w+', shape=(total_frames, 3, 5, 256))
    frame_buffer = []
    def batch_encode(frame_list : List[np.ndarray]):
        nonlocal num_frame_processed
        nonlocal num_batch_processed
        clip_frames = len(frame_list)
        for _ in range(frame_batch_size - len(frame_list)):
            frame_list.append(np.zeros_like(frame_list[0]))
        frames = np.stack(frame_list)
        clip = frames / 255.0
        clip = jnp.array(clip, dtype=jnp.bfloat16)
        vae_clip, _ = encode_video(vae_params, clip)
        vae_clip = vae_clip.astype(np.float32)
        vae_video_np[num_frame_processed:num_frame_processed+clip_frames] = vae_clip if clip_frames == frame_batch_size else vae_clip[:clip_frames]
        vae_video_np.flush()
        num_frame_processed += clip_frames
        num_batch_processed += 1
        frame_list.clear()
        
        return np.array(vae_clip), clip_frames
    for frame in video.iter_frames():
        frame_buffer.append(frame)
        if len(frame_buffer) == frame_batch_size:
            batch_encode(frame_buffer)
    assert num_batch_processed == total_batch
    if len(frame_buffer) > 0:
        assert len(frame_buffer) == remained_num_frame and remained_num_frame < frame_batch_size
        batch_encode(frame_buffer)
    assert num_frame_processed == total_frames
    return vae_video_np

from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_video(video_path: str, output_dir: str):
    """
    Function to process a single video. This is the task run by each thread.
    """
    try:
        base = os.path.basename(os.path.splitext(video_path)[0])
        done_path = os.path.join(output_dir, f"{base}.done")
        output_path = os.path.join(output_dir, f"{base}.vae")

        if os.path.exists(done_path):
            print(f"Skipping {video_path}")
            return f"Skipped {video_path}"

        print(f"Working on {video_path}")
        
        # NOTE: VideoFileClip creation can be resource-intensive. 
        # For pure CPU-bound tasks, you might consider multiprocessing.
        # For I/O-heavy work (like saving the file), threading is fine.
        # Given this uses moviepy, which relies on libraries like ffmpeg (often C/C++),
        # threading might still provide a performance boost by managing I/O concurrently.
        video = VideoFileClip(str(video_path))
        
        # Execute the actual encoding
        encode_one_video(video, output_path)
        
        # Close the clip to release resources
        video.close() 

        with open(done_path, 'w') as f:
            f.write("done")
        
        return f"Successfully processed {video_path}"

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return f"Error processing {video_path}: {e}"

## Multithreaded Version
def encode_all_videos_multithreaded(video_paths: List[str], output_dir: str, max_workers: int = 8):
    """
    Converts and encodes videos using a ThreadPoolExecutor for concurrency.
    
    :param video_paths: A list of paths to the video files.
    :param output_dir: The directory where the encoded videos and .done flags will be saved.
    :param max_workers: The maximum number of threads to use.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each video processing task to the executor
        future_to_video = {
            executor.submit(process_single_video, video_path, output_dir): video_path
            for video_path in video_paths
        }
        
        # Wait for all futures to complete and print results
        for future in tqdm(as_completed(future_to_video), total=len(video_paths)):
            video_path = future_to_video[future]
            try:
                result = future.result()
                # The 'result' is the string returned by process_single_video
                # The process_single_video function already prints errors/skips/starts
                if not result.startswith(("Error", "Skipped")):
                    print(f"--- Task for {video_path} finished ---")
            except KeyboardInterrupt:
                print(f"Interrupted")
                return
            except Exception as exc:
                # This catches exceptions from the executor itself, 
                # though most task-specific errors are handled inside process_single_video
                print(f'{video_path} generated an exception: {exc}')

import glob
root_path = "/home/zzhang18/nchen3/neo_video_generation/vpt_process/new_local_video"
vae_path = "/home/zzhang18/nchen3/neo_video_generation/vpt_process/vae_video"
video_paths = glob.glob(root_path + "/*.mp4")
os.makedirs(vae_path, exist_ok=True)
encode_all_videos_multithreaded(video_paths, vae_path, max_workers=16)