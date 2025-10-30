import json
import os
import requests
from tqdm.autonotebook import tqdm
import requests
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# NOTE: download_and_write remains the same, but imports requests, os, and tqdm
def download_and_write(url: str, filename: str):
    """Downloads a file from a URL and writes it to a local filename."""
    # Ensure the directory exists before attempting to open the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with requests.get(url, stream=True, timeout=10) as r:
        r.raise_for_status()
        
        # Use a descriptive title for tqdm for better progress tracking
        file_base = os.path.basename(filename)
        
        with open(filename, 'wb') as f:
            # 1Mb chunk size
            # Note: TQDM is typically less effective/harder to manage in parallel processing,
            # but we keep it here for the single-file progress if run directly.
            # In the parallel version, overall progress is tracked with as_completed.
            for chunk in r.iter_content(chunk_size=1024 * 2**10):
                f.write(chunk)

# --- New Wrapper Function ---
def _download_task(basedir: str, mp4_relpath: str, action_relpath: str, video_folder: str):
    """
    Handles the complete download logic for a single video/action pair.
    Designed to be run by the ThreadPoolExecutor.
    """
    mp4_filename = os.path.join(video_folder, os.path.basename(mp4_relpath))
    action_filename = os.path.join(video_folder, os.path.basename(action_relpath))
    
    # The done file should use the full path to the video file's name without extension
    video_base_name = os.path.basename(os.path.splitext(mp4_relpath)[0])
    done_file = os.path.join(video_folder, video_base_name) + '.done'

    if os.path.exists(done_file):
        # Return a message instead of printing, so the main thread can log it
        return f"Skipping {mp4_relpath} (already done)"

    # Print before starting
    print(f"Starting download for {mp4_relpath}")
    
    try:
        # 1. Download MP4
        download_and_write(basedir + mp4_relpath, mp4_filename)
        
        # 2. Download JSONL
        download_and_write(basedir + action_relpath, action_filename)
        
        # 3. Create a done file (only on success)
        with open(done_file, 'w') as f:
            f.write("done")
            
        return f"Successfully downloaded and marked as done: {mp4_relpath}"

    except Exception as e:
        # Return error message to the main thread
        return f"ERROR downloading {mp4_relpath}: {e}"

# --- Parallelized Main Function ---
def download_json_parallel(video_folder: str, json_path: str = './build-house-Jul-28.json'):
    # Load the JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    basedir = data['basedir']
    mp4_relpaths = data['relpaths']
    action_relpaths = [os.path.splitext(path)[0] + '.jsonl' for path in mp4_relpaths]
    
    # Create the target directory if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Use a ThreadPoolExecutor for concurrent I/O-bound tasks
    # Max workers set to 8 as a safe, reasonable limit for I/O
    MAX_WORKERS = 8 
    print(f"Starting download of {len(mp4_relpaths)} tasks with up to {MAX_WORKERS} concurrent threads...")
    
    # Store future objects to track progress
    futures = []
    
    # 1. Submit all tasks to the pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for mp4_relpath, action_relpath in zip(mp4_relpaths, action_relpaths):
            future = executor.submit(
                _download_task, 
                basedir, 
                mp4_relpath, 
                action_relpath, 
                video_folder
            )
            futures.append(future)

        # 2. Wait for all tasks to complete and process results
        # Use as_completed for dynamic reporting as tasks finish
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall Download Progress"):
            # Get the return value from the _download_task function
            result_message = future.result()
            print(result_message)
            
    print("All download tasks complete.")
urls = ["https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/waterfall-Jul-28.json",
        "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/pen-animals-Jul-28.json",
        "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/build-house-Jul-28.json",
        "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/find-cave-Jul-28.json"]
root = "/home/zzhang18/nchen3/neo_video_generation/vpt_process/"
video_folder = os.path.join(root, "new_local_video")
json_folder = os.path.join(root, "video_json")
os.makedirs(video_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)
for url in urls:
    base = os.path.basename(url)
    json_path = os.path.join(json_folder, base)
    if not os.path.exists(json_path):
        download_and_write(url = url, filename = json_path)
for url in urls:
    base = os.path.basename(url)
    json_path = os.path.join(json_folder, base)
    while True:
        try:
            download_json_parallel(video_folder, json_path)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Timeout, retry")
            print(e)
            continue
        break