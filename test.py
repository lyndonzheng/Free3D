import os
import argparse
import torch
import math
import numpy as np

from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from models.ddim import DDIMSampler
from utils.util import instantiate_from_config, write_video, pred_bbox, image_preprocess_nosave
from utils.sam_utils import sam_init, sam_out_nosave
from utils import camera


def tensor2image(img_tensor):
    grid = img_tensor.detach().cpu()
    grid = torch.clamp(grid, -1., 1.)
    grid = (grid + 1.0) / 2.0
    grid = rearrange(grid, 'b c h w-> b h w c')
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    return grid


def preprocess_image(models, img_path):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''
    # preprocess image
    img = Image.open(img_path)
    print('old input_im:', img.size)
    if img.mode == 'RGBA':
        img = np.array(img, dtype=np.float32) / 255.0
        img = img[:, :, 3:4] * img+ (1.0 - img[:, :, 3:4]) * np.ones_like(img)
        img = img[:, :, :3]
    else:
        img.thumbnail([512, 512], Image.Resampling.LANCZOS)
        image_sam = sam_out_nosave(models['sam'], img.convert("RGB"), pred_bbox(img))
        input_256 = image_preprocess_nosave(image_sam, lower_contrast=False, rescale=True)
        torch.cuda.empty_cache()
        img = np.array(input_256, dtype=np.float32) / 255.0
    print('new input_im:', img.size)

    return img


def get_sample_ray(extrinsics, intrinsics=None, img_h=32, img_w=32):
    '''
    given camera extrinsics and intrinsics, sampling the camera ray
    '''
    if intrinsics is None:
        focal = .5 / np.tan(.5 * 0.8575560450553894)
        intrinsics = torch.tensor(np.array([[focal, 0.0, 1.0 / 2.0],
                                            [0.0, focal, 1.0 / 2.0],
                                            [0.0, 0.0, 1.0]
                                            ]).astype(np.float32)).unsqueeze(0).repeat(extrinsics.size(0), 1, 1)
    
    # get relative pose
    first_c2w = torch.linalg.inv(extrinsics[0,...])
    rel_w2c = torch.einsum('bnm,bmp->bnp', extrinsics, first_c2w.unsqueeze(0).repeat(extrinsics.size(0),1,1))
    extrinsics = rel_w2c

    intrinsics[:, 0, :] *= img_h
    intrinsics[:, 1, :] *= img_w
    centers, rays = camera.get_center_and_ray(img_h=img_h, img_w=img_w, pose=extrinsics[:,:3,:],
                                              intr=intrinsics, device=extrinsics.device, legacy=False)
    
    # plucker embedding
    norm_ray = torch.nn.functional.normalize(rays, dim=-1)
    plucker_ray = torch.cat((norm_ray, torch.cross(centers, norm_ray, dim=-1)), dim=-1)
    return plucker_ray


def main_run(models, device, return_what, src_path, x=0.0, y=0.0, z=0.0, 
             N_views=10, n_samples=1, scale=3.0, save_path = None, ddim_steps=50, 
             ddim_eta=1.0, precision='fp32', h=256, w=256, fname=''):
    
    src_img = preprocess_image(models, src_path)
    src_save_path = os.path.join(save_path, fname+'_src.png')
    Image.fromarray((src_img*255).astype(np.uint8)).save(src_save_path)

    # get input pose
    if 'image' in return_what:
        N_views = 2
        T = torch.tensor([[0, math.sin(0), math.cos(0), 0],
                [math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), math.radians(z)]])
        w2cs = camera.spherical_to_camera(torch.tensor([[math.radians(90)],[math.radians(90+x)]]), torch.tensor([[0],[math.radians(y)]]), torch.tensor([[2.0],[z+2.0]]))
        plucker_ray = get_sample_ray(w2cs)
    elif "video" in return_what:
        theta = torch.linspace(0, 0, N_views + 1).unsqueeze(1)
        azimuth = (torch.linspace(0, 2.0 * np.pi, N_views + 1).unsqueeze(1)) % (2.0 * np.pi)
        z = torch.tensor([2.0]).repeat(N_views + 1).unsqueeze(1)
        T = torch.cat([theta-theta[0:1,:], torch.sin(azimuth), torch.cos(azimuth), z-z[0:1,:]], dim=-1)[:N_views,:]
        w2cs = camera.spherical_to_camera(theta, azimuth, z)[:N_views]
        plucker_ray = get_sample_ray(w2cs)

    # get input image
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
    src_img = src_img * 2 - 1
    src_img = transforms.functional.resize(src_img, [h, w])
    
    sampler = DDIMSampler(models['free3d'])
    precision_scope = torch.autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with models['free3d'].ema_scope():
            # get conditional input
            src_concat = models['free3d'].encode_first_stage(src_img).mode().detach().unsqueeze(1).repeat(n_samples, N_views, 1, 1, 1)
            src_concat = rearrange(src_concat, 'b v c h w -> (b v) c h w')
            # get cross attention condition
            src_cross = models['free3d'].get_learned_conditioning(src_img).unsqueeze(1).repeat(n_samples, N_views, 1, 1)
            T = T[None, :].to(src_cross.device).repeat(n_samples, 1, 1).unsqueeze(2)
            src_cross = rearrange(torch.cat([src_cross, T], dim=-1), 'b v l c -> (b v) l c')
            src_cross = models['free3d'].cc_projection(src_cross)
            # get ray conditioning 
            pose_emb = models['free3d'].ray_embedding(plucker_ray.to(device)).unsqueeze(1).repeat(n_samples, 1, 1, 1)
            pose_emb = rearrange(pose_emb, 'b v (h w) c-> (b v) c h w', h=32, w=32)
            cond = {}
            cond['c_crossattn'] = [src_cross]
            cond["c_concat"] = [src_concat]
            cond["c_pose"] = pose_emb

            if scale != 1.0:
                uc = {}
                uc["c_crossattn"] = [torch.zeros_like(src_cross).to(device)]
                uc["c_concat"] = [torch.zeros_like(src_concat).to(device)]
                uc["c_pose"] = torch.zeros_like(pose_emb).to(device)
            else:
                uc = None

            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                batch_size=src_concat.shape[0],
                shape=[4, h // 8, w // 8],
                conditioning=cond,
                eta=ddim_eta,
                temperature=0.3,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                repeat_noise=True,
            )
            print(samples_ddim.shape)
            x_samples = models['free3d'].decode_first_stage(samples_ddim)

            grid = tensor2image(x_samples)
            grid = rearrange(grid, '(b v) ... -> b v ...', b=n_samples)
            for i in range(grid.shape[0]):
                if 'video' in return_what:
                    filename = fname + '_results_' + str(i) + '.mp4'
                    out_save_path = os.path.join(save_path, filename)
                    os.makedirs(os.path.split(out_save_path)[0], exist_ok=True)
                    write_video(out_save_path, grid[i])
                for j in range(N_views):
                    filename = fname + '_results_' + str(j) + '.png'
                    out_save_path = os.path.join(save_path, filename)
                    os.makedirs(os.path.split(out_save_path)[0], exist_ok=True)
                    Image.fromarray(grid[i][j]).save(out_save_path)


def run_demo():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help='the gpu num')
    parser.add_argument('--resume', type=str, default='./checkpoints/', help='the pre-trained checkpoint')
    parser.add_argument('--sam_path', type=str, default='', help='the pre-trained sam checkpoint')
    parser.add_argument('--config', type=str, default='configs/free3d_test.yaml')
    parser.add_argument('--img_path', type=str, default='./examples/one2-3-45', help='render the target images or videos')
    parser.add_argument('--gen_type', type=str, default='video', help='render the target images or videos')
    parser.add_argument('--views', type=int, default=50, help='the generated views num')
    parser.add_argument('--save_path', type=str, default='./results', help='save path for the results')

    opt = parser.parse_args()

    device = f"cuda:{opt.gpu}"
    config = OmegaConf.load(opt.config)
    models = dict()
    seed_everything(42)
    # load stable diffusion model
    print('Instantiating LatentDiffusion...')
    config['model']['params']['ckpt_path'] = opt.resume
    models['free3d'] = instantiate_from_config(config.model)
    models['free3d'].to(device)
    models['free3d'].eval()
    # background removal model
    print('Instantiating SAM model...')
    models['sam'] = sam_init(opt.gpu, opt.sam_path)
    
    # image path
    os.makedirs(opt.save_path, exist_ok=True)
    fname = opt.img_path.split('/')[-1].split('.')[0]
    main_run(models, device, opt.gen_type, opt.img_path, x=0, y=90, z=0, 
             N_views=opt.views, n_samples=1, save_path = opt.save_path, fname=fname)
    

if __name__ == '__main__':
    run_demo()