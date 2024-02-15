import os
import argparse
import torch
import numpy as np

from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils.util import instantiate_from_config, write_video


def tensor2image(img_tensor):
    grid = img_tensor.detach().cpu()
    grid = torch.clamp(grid, -1., 1.)
    grid = (grid + 1.0) / 2.0
    grid = rearrange(grid, 'b c h w-> b h w c')
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    return grid

def run_demo():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=int, default=0, help='the gpu num')
    parser.add_argument('--resume', type=str, default='', help='the pre-trained checkpoint')
    parser.add_argument('--config', type=str, default='configs/objaverse.yaml')
    parser.add_argument('--gen_type', type=str, default='images', help='render the target images or videos')
    parser.add_argument('--save_path', type=str, default='', help='save path for the results')

    opt = parser.parse_args()

    device = f"cuda:{opt.gpus}"
    config = OmegaConf.load(opt.config)
    models = dict()
    seed_everything(42)
    # load stable diffusion model
    print('Instantiating LatentDiffusion...')
    config['model']['params']['ckpt_path'] = opt.resume
    model = instantiate_from_config(config.model)
    model.to(device)
    model.eval()
    # load the dataset
    print('Preparing to load dataset on gpu')
    config['data']['params']['batch_size'] = 32
    config['data']['params']['num_workers'] = 32
    data = instantiate_from_config(config.data)
    # set the evaluation dataset
    # only the objaverse dataset is used for training
    # all images in gso and omniobject is used for evaluation
    if  config['data']['params']['dataname'] == 'objaverse':
        test_data = data.val_dataloader()
    else:
        test_data = data.train_dataloader() 
    # save path
    root = os.path.join(opt.save_path, opt.resume.split('/')[-3])
    for data in test_data:

        # render the target images given pose
        if 'images' in opt.gen_type:
            images = model.log_images(data, N=data['images'].size()[0], n_row=data['images'].size()[0],
                                      ddim_steps=50, inpaint=True, plot_progressive_rows=False, plot_diffusion_rows=False,
                                      unconditional_guidance_scale=3.0, unconditional_guidance_label=[""], use_ema_scope=False)
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    if images[k].size(0) == data['images'].size()[0]:
                        grid = tensor2image(images[k])
                    else:
                        grid = tensor2image(rearrange(images[k], '(b v) ... -> b v ...', v=config.model.params.unet_config.params.views)[:,1,...])
                    for i in range(grid.shape[0]):
                        filename = k + '/' + data['filename'][i] + '.png' 
                        path = os.path.join(root, filename)
                        os.makedirs(os.path.split(path)[0], exist_ok=True)
                        Image.fromarray(grid[i]).save(path)

        # render a video given src images
        n_frames = 50
        if 'videos' in opt.gen_type:
            images = model.log_videos(data, N=data['images'].size()[0], n_row=data['images'].size()[0],
                                      ddim_steps=50,inpaint=True, plot_progressive_rows=False, plot_diffusion_rows=False,
                                      unconditional_guidance_scale=3.0, unconditional_guidance_label=[""], use_ema_scope=False,
                                      model='circle', n_frames=n_frames)

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    grid = tensor2image(images[k])
                    grid = rearrange(grid, '(b v) ... -> b v ...', b = data['images'].size()[0])
                    for i in range(grid.shape[0]):
                        filename = k + '/' + data['filename'][i] + '.mov'
                        path = os.path.join(root, filename)
                        os.makedirs(os.path.split(path)[0], exist_ok=True)
                        write_video(path, grid[i]) 

if __name__ == '__main__':
    run_demo()