import torch
@torch.no_grad
def display_mnist_video_tensor(video, fps=4):
    import wandb
    # Ensure tensor is on CPU
    # B, C, T, H, W
    # 4, 3, 16, 64, 64
    # the input should be in [0, 1] !
    if video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1, 1)
    video = torch.clip(video, 0, 1).transpose(1,2)
    video = (video * 255).to(torch.uint8)
    video = video.cpu().detach().numpy()
    # Display the video
    video = wandb.Video(video, fps=fps, format="gif")
    wandb.log({"video": video}, commit=False)