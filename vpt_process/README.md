To generate vae files, first run `python video_downloader.py` to download videos. 
Then run `python vae_preprocess.py to convert video to vae`.
Next run `python invalid_vae.py` to detect vaes that contain nan.
Finally run `python cleanup_nan.py` to remove nan videos.