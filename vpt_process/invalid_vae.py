import glob
import numpy as np
vae_files = glob.glob("./vae_video/*.vae")
invalid_files = []
for file in vae_files:
    _m = np.memmap(file, mode="r", dtype=np.float32).reshape(-1, 3, 5, 256)
    if np.isnan(_m).any():
        invalid_files.append(file)
        print("invalid", file)
    else:
        print("pass", file)
with open("./vae_video/invalid_doc", 'w') as f:
    for file in invalid_files:
        print(file, file = f)