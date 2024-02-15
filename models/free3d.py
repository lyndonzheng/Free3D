import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import ListConfig
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from torch.optim.lr_scheduler import LambdaLR
from models.ddpm import LatentDiffusion, exists
from utils.util import instantiate_from_config, count_params, exists, default, log_txt_as_img, isimage, ismap
from utils.camera import cartesian_to_spherical, get_center_and_ray, spherical_to_camera, get_interpolate_render_path


class Free3DDiffusion(LatentDiffusion):
    '''
    Free3D: Consistent Novel View Synthesis without 3D Representation
    '''
    def __init__(self, 
                 concat_keys=('image', 'ray'),
                 ray_embedding_config=None,
                 *args, 
                 **kwargs):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", list())
        super().__init__(*args, **kwargs)
        self.tgt_num = kwargs['unet_config']['params']['views']
        self.concat_channel = kwargs['unet_config']['params']['in_channels'] - kwargs['unet_config']['params']['out_channels']
        self.use_3d_transformer = kwargs['unet_config']['params']['use_3d_transformer']
        self.use_global_conditional = kwargs['unet_config']['params']['use_global_conditional']
        # construct linear projection layer for concatenating image CLIP embedding and RT
        self.cc_projection = nn.Linear(772, 768) if self.use_global_conditional else nn.Linear(768, 768)
        nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
        nn.init.zeros_(list(self.cc_projection.parameters())[1])
        self.cc_projection.requires_grad_(True)
        self.ray_embedding = instantiate_from_config(ray_embedding_config)

        self.concat_keys = concat_keys

        if exists(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_keys)


    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        self.params_new = []

        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                new_shape = param.shape
                if not name in sd: 
                    if 'diffusion' in name:
                        print(f"Manual zero init:{name} with new shape {new_shape} ")
                        new_param = param.clone().zero_()
                        sd[name] = new_param
                        self.params_new.append(name)
                    else:
                        continue
                old_shape = sd[name].shape
                assert len(old_shape)==len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    print(f"Manual init:{name} with new shape {new_shape} and old shape {old_shape}")
                    new_param = param.clone().zero_()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        index_size = min(new_param.shape[0], old_param.shape[0])
                        new_param[:index_size] = old_param[:index_size]
                    elif len(new_shape) >= 2:
                        index_o_size = min(new_param.shape[0], old_param.shape[0])
                        index_i_size = min(new_param.shape[1], old_param.shape[1])
                        new_param[:index_o_size, :index_i_size] = old_param[:index_o_size, :index_i_size]

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_views=1,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False, uncond=0.05, train=True):
        """get the conditional input"""
        images = rearrange(batch['images'].to(memory_format=torch.contiguous_format).float(), 'b v h w c -> b v c h w').to(self.device)
        extrinsics = batch['w2cs'].to(memory_format=torch.contiguous_format).float().to(self.device)
        intrinsics = batch['intrinsics'].to(memory_format=torch.contiguous_format).float().to(self.device)

        if bs is not None:
            images = images[:bs]
            extrinsics = extrinsics[:bs]
            intrinsics = intrinsics[:bs]

        self.b, v, _, self.h, self.w = images.size()
        # get encoded features
        src_img = images[:, 0, ...]
        # get target image
        index = [i for i in range(v)]
        tgt_index = index[:self.tgt_num] if self.use_3d_transformer else index[1:self.tgt_num+1]
        tgt_img = rearrange(images[:, tgt_index, ...], 'b v ...-> (b v) ...')
        encoder_posterior = self.encode_first_stage(tgt_img)
        tgt_z = self.get_first_stage_encoding(encoder_posterior).detach()
        # get global camera information
        src_cam = rearrange(repeat(extrinsics[:, 0:1, ...], 'b 1 ...-> b v ...', v=self.tgt_num), 'b v ... -> (b v) ...')
        tgt_cam = rearrange(extrinsics[:, tgt_index, ...], 'b v ... -> (b v) ...')
        T = self.get_global_relative_cam(src_cam, tgt_cam)
        # get local camera information
        src_c2w = torch.linalg.inv(extrinsics[:, 0,...])
        rel_w2c = torch.einsum('bvnm,bvmp->bvnp', extrinsics, src_c2w.unsqueeze(1).repeat(1,v,1,1))
        extrinsics = rel_w2c
        img_h, img_w = 32, 32
        intrinsics[:,:,0,:] *= img_h
        intrinsics[:,:,1,:] *= img_w
        centers, rays = get_center_and_ray(img_h=img_h, img_w=img_w, pose=extrinsics.flatten(0,1)[:,:3,:], 
                                               intr=intrinsics.flatten(0,1), device=extrinsics.device, legacy=False)
        centers = rearrange(centers, '(b v) n c -> b v n c', b=self.b)
        rays = rearrange(rays, '(b v) n c -> b v n c', b=self.b)
        # plucker w/ position embedding
        tgt_ray = torch.nn.functional.normalize(rays[:, tgt_index, ...], dim=-1)
        tgt_pose = torch.cat((tgt_ray, torch.cross(centers[:, tgt_index,...], tgt_ray, dim=-1)), dim=-1)
        tgt_emb = self.ray_embedding(tgt_pose)
        tgt_emb = rearrange(tgt_emb, 'b v (h w) c-> (b v) c h w', h=img_h, w=img_w)
        # get conditional information
        cond = {}
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        random = torch.rand(tgt_z.size(0), device=tgt_z.device)
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        null_prompt = self.get_learned_conditioning([""]).detach()
        # get cross and concanate conditional information
        src_z = self.encode_first_stage(src_img).mode().detach()
        src_z = rearrange(repeat(src_z.unsqueeze(1), 'b 1 ... -> b v ...', v=self.tgt_num), 'b v ... -> (b v) ...')
        if train:
            with torch.enable_grad():    
                clip_emb = self.get_learned_conditioning(src_img).detach()
                clip_emb = rearrange(repeat(clip_emb.unsqueeze(1), 'b 1 ... -> b v ...', v=self.tgt_num), 'b v ... -> (b v) ...')
                null_prompt = self.get_learned_conditioning([""]).detach()
                if self.use_global_conditional:
                    cond["c_crossattn"] = [self.cc_projection(torch.cat([torch.where(prompt_mask, null_prompt, clip_emb), T[:, None, :]], dim=-1))]
                else:
                    cond["c_crossattn"] = [self.cc_projection(torch.where(prompt_mask, null_prompt, clip_emb))]
                cond["c_concat"] = [input_mask * torch.cat([src_z], dim=1)]
                cond["c_pose"] = input_mask * tgt_emb
        else:
            clip_emb = self.get_learned_conditioning(src_img).detach()
            clip_emb = rearrange(repeat(clip_emb.unsqueeze(1), 'b 1 ... -> b v ...', v=self.tgt_num), 'b v ... -> (b v) ...')
            if self.use_global_conditional:
                cond["c_crossattn"] = [self.cc_projection(torch.cat([clip_emb, T[:, None, :]], dim=-1))]
            else:
                cond["c_crossattn"] = [self.cc_projection(clip_emb)]
            cond["c_concat"] = [torch.cat([src_z], dim=1)]
            cond["c_pose"] = tgt_emb

        out = [tgt_z, cond]
        if return_first_stage_outputs:
            tgt_rec = self.decode_first_stage(tgt_z)
            out.extend([tgt_img, tgt_rec])
        if return_original_cond:
            out.append(src_img)
        return out
    

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True, x_T=None, repeat_noise=False, **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N,
                                           train=False)
        N = x.shape[0]
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x.detach().cpu()
        log["reconstruction"] = xrec.detach().cpu()
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc.detach().cpu()
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2]//25)
                log["conditioning"] = xc.detach().cpu()
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2]//25)
                log['conditioning'] = xc.detach().cpu()
            elif isimage(xc):
                log["conditioning"] = xc.detach().cpu()
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc).detach().cpu()

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label, image_size=x.shape[-1])
            # uc = torch.zeros_like(c)
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc, repeat_noise=repeat_noise,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg.detach().cpu()

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        
        return log


    @torch.no_grad()
    def get_global_relative_cam(self, src_cam, tgt_cam):
        """get relative camera pose in spherical space"""
        src_theta, src_azimuth, src_z = cartesian_to_spherical(src_cam[:,:3,:3], src_cam[:,:3,-1])
        tgt_theta, tgt_azimuth, tgt_z = cartesian_to_spherical(tgt_cam[:,:3,:3], tgt_cam[:,:3,-1])
        theta = tgt_theta - src_theta
        azimuth = (tgt_azimuth - src_azimuth) % (2 * torch.pi)
        z = tgt_z - src_z
        global_camera_rel = torch.cat([theta, torch.sin(azimuth), torch.cos(azimuth), z], dim=1)
        return global_camera_rel

    def forward(self, x, c, *args, **kwargs):
        """
        For the same instance, add the same scale noise for pseudo 3D attention
        """
        t = rearrange(repeat(torch.randint(0, self.num_timesteps, (self.b, 1), device=self.device).long(), 'b 1 -> b v', v=self.tgt_num), 'b v -> (b v)')
        return self.p_losses(x, c, t, *args, **kwargs)
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None, image_size=512):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            raise NotImplementedError("todo")
        
        c = c.repeat(batch_size, 1, 1).to(self.device)
        cond = {}
        cond["c_crossattn"] = [c]
        cond["c_concat"] = [torch.zeros([batch_size, self.concat_channel, image_size // 8, image_size // 8]).to(self.device)]
        img_size = 32
        cond["c_pose"] = torch.zeros([batch_size, self.ray_embedding.get_output_dim(6), img_size, img_size]).to(self.device)

        return cond
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params_old = []
        params_new = []
        for name, params in self.model.named_parameters():
            if 'model.'+name in self.params_new:
                params_new.append(params)
            else:
                params_old.append(params)
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params_old.append(self.logvar)
        opt = torch.optim.AdamW([{'params': params_old, 'lr': 1.0 * lr},
                                 {'params': params_new, 'lr': 10.0 * lr},
                                 {'params': self.cc_projection.parameters(), 'lr': 1.0 * lr},
        ], lr = lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt