import os
with open("vae_video/invalid_doc") as f:
    for i in f.readlines():
        i = i.strip()
        if os.path.exists(i):
            print(f"Removing {i}")
            os.remove(i)
        else:
            print(f"The file {i} is removed already")