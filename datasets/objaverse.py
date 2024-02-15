import os
import sys
import glob
import json
import math
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from pathlib import Path
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset


from utils.vis_camera import vis_points_images_cameras

class ObjaverseDataset(Dataset):
    def __init__(self,
                 root_dir = '',
                 cfg = None,
                 debug = False,
                 ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.total_view = cfg.total_view
        self.load_view = cfg.load_view
        self.debug = debug

        # load the file name
        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
        # split the dataset for training and validation
        total_objects = len(self.paths)
        self.validation = cfg.validation
        if self.validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))

        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)
        
        image_transforms = [
            torchvision.transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ]
        self.image_transforms = transforms.Compose(image_transforms)

    def __len__(self):
        return len(self.paths)
    
    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            os.remove(path)
            os.remove(path.replace('png', 'npy'))
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
    def pre_data(self, paths, views):
        '''
        load the data for given filename 
        '''
        color = [1., 1., 1., 1.]
        imgs = []
        w2cs = []
        intrinsics = []
        index = range(views) if self.validation else torch.randperm(views)
        # # find the closer views
        for i in index[:self.load_view]:
            img = self.process_img(self.load_im(paths[i], color)).unsqueeze(0)
            imgs.append(img)
            w2c_gl = np.load(paths[i].replace('png', 'npy'))
            w2cs.append(w2c_gl)
            focal = .5 / np.tan(.5 * 0.8575560450553894)
            intrinsics.append(np.array([[focal, 0.0, 1.0 / 2.0],
                                        [0.0, focal, 1.0 / 2.0],
                                        [0.0, 0.0, 1.0]]))
        imgs = torch.cat(imgs)
        intrinsics = torch.tensor(np.array(intrinsics)).to(imgs)
        w2cs = torch.tensor(np.array(w2cs)).to(imgs)
        w2cs_gl = torch.eye(4).unsqueeze(0).repeat(imgs.size(0),1,1)
        w2cs_gl[:,:3,:] = w2cs
        # camera poses in .npy files are in OpenGL convention: 
        #     x right, y up, z into the camera (backward),
        # need to transform to COLMAP / OpenCV:
        #     x right, y down, z away from the camera (forward)
        w2cs = torch.einsum('nj, bjm-> bnm', self.opengl_to_colmap, w2cs_gl)
        c2ws = torch.linalg.inv(w2cs)
        camera_centers = c2ws[:, :3, 3].clone()
        # fix the distance of the source camera to the object / world center
        assert torch.norm(camera_centers[0]) > 1e-5
        translation_scaling_factor = 2.0 / torch.norm(camera_centers[0])
        w2cs[:, :3, 3] *= translation_scaling_factor
        c2ws[:, :3, 3] *= translation_scaling_factor
        camera_centers *= translation_scaling_factor
        return imgs, w2cs, c2ws, intrinsics
    
    def __getitem__(self, index):
        # load the rendered images
        try:
            filename = os.path.join(self.root_dir, self.paths[index])
            paths = glob.glob(filename + '/*.png')
            views = len(paths)
            imgs, w2cs, c2ws, intrinsics = self.pre_data(paths, views)
        except:
            filename = os.path.join(self.root_dir, '8e348d4d2f2949cf88bd896a92a4364d')
            paths = glob.glob(filename + '/*.png')
            views = len(paths)
            imgs, w2cs, c2ws, intrinsics = self.pre_data(paths, views)

        # debug the camera system for debugging
        if self.debug:
            import pdb
            pdb.set_trace()
            intrinsics[:, 0, :] *=256
            intrinsics[:, 1, :] *=256
            vis_points_images_cameras(w2cs, intrinsics, imgs, frustum_size=0.5, filename=filename.split('/')[-1] + 'camemra_ori.html')

        data = {
                'images': imgs,
                'w2cs': w2cs,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'filename': filename.split('/')[-1]
        }
        return data

    def process_img(self, img):
        img = img.convert("RGB")
        return self.image_transforms(img)