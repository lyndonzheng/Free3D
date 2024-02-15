import os
import sys
import json
import torch
import torchvision
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from PIL import Image
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


from utils.vis_camera import vis_points_images_cameras


class OmniObject3DDataset(Dataset):
    def __init__(self,
                 root_dir = '',
                 cfg = None,
                 debug = False,
                 ):
        super().__init__()

        self.base_path = root_dir
        self.load_view = cfg.load_view
        self.stage = cfg.stage
        self.debug = debug
        assert os.path.exists(self.base_path)
        assert self.stage in ['train', 'val', 'test'], 'mode must be either "train", "val" or "test"!'
        all_objs = []

        if self.stage == 'train':
            cats = sorted(os.listdir(self.base_path))  # default: train on all the categories
            # cats = ['toy_train', 'bread', 'cake', 'toy_boat', 'hot_dog', 'wallet', 'pitaya', 'squash', 'handbag', 'apple'] # change cats for individual needs
            for cat in cats:
                scans = sorted(os.listdir(os.path.join(self.base_path, cat)))
                scans = scans[3:]
                objs = [(cat, os.path.join(self.base_path, cat, x)) for x in scans]
                all_objs.extend(objs)
        if self.stage == 'val':
            cats = sorted(os.listdir(self.base_path)) # default: train on all the categories
            # cats = ['toy_train', 'bread', 'cake', 'toy_boat', 'hot_dog', 'wallet', 'pitaya', 'squash', 'handbag', 'apple'] # change cats for individual needs
            for cat in cats:
                scans = os.listdir(os.path.join(self.base_path, cat))
                scans.sort()
                scans = scans[:3]
                objs = [(cat, os.path.join(self.base_path, cat, x)) for x in scans]
                all_objs.extend(objs)
        elif self.stage == 'test':
            cat = os.path.basename(self.base_path)
            scans = os.listdir(os.path.join(self.base_path))
            scans.sort()
            objs = [(cat, os.path.join(self.base_path, x)) for x in scans]
            all_objs.extend(objs)
        elif self.stage == 'testsub':
            scans = sorted(os.listdir(self.base_path))
            objs = [('category_agnostic', os.path.join(self.base_path, x)) for x in scans]
            all_objs.extend(objs)

        self.all_objs = all_objs

        self._coord_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        print("Loading OO3D dataset", self.base_path, "stage", self.stage, len(self.all_objs), "objs",)

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
        return len(self.all_objs)
    
    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = imageio.imread(path)[...,:3]
            img[img.sum(axis=-1) == 0] = 255
            img = Image.fromarray(img)
        except:
            print(path)
            sys.exit()
        return img
    
    def _pre_data(self, basedir):
        '''
        load the data for give filename
        '''
        color = [1., 1., 1., 1.]
        imgs = []
        img_files = []
        w2cs = []
        intrinsics = []
        with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)

        # load the image and camera parameters
        camera_angle_x = float(meta['camera_angle_x'])
        index = range(len(meta['frames'])) #torch.randperm(len(meta['frames'])) if self.stage =='train' else range(len(meta['frames']))
        for i in index[:self.load_view]:
            frame = meta['frames'][i]
            fname = os.path.join(basedir, 'images', frame['file_path'].split('/')[-1] + '.png')
            img_files.append(fname)
            img = self.process_img(self.load_im(fname, color)).unsqueeze(0)
            imgs.append(img)
            c2w_gl = np.array(frame['transform_matrix'])
            w2c_gl = np.linalg.inv(c2w_gl)
            w2cs.append(w2c_gl)
            focal = .5 / np.tan(.5 * camera_angle_x)
            intrinsics.append(np.array([[focal, 0, 1.0 / 2.0],
                                         [0, focal, 1.0 / 2.0 ],
                                         [0, 0, 1],
            ]))

        imgs = torch.cat(imgs)
        intrinsics = torch.tensor(np.array(intrinsics)).to(imgs)
        w2cs_gl = torch.tensor(np.array(w2cs)).to(imgs)
        # camera poses in .npy files are in OpenGL convention: 
        #     x right, y up, z into the camera (backward),
        # need to transform to COLMAP / OpenCV:
        #     x right, y down, z away from the camera (forward)
        w2cs = torch.einsum('nj, bjm-> bnm', self.opengl_to_colmap, w2cs_gl)
        c2ws = torch.linalg.inv(w2cs)


        return imgs, w2cs, c2ws, intrinsics

    def __getitem__(self, index):

        cat, data_dir = self.all_objs[index]
        data_dir = os.path.join(data_dir, 'render')
        imgs, w2cs, c2ws, intrinsics = self._pre_data(data_dir)

        # debug the camera system for debugging
        if self.debug:
            import pdb
            pdb.set_trace()
            intrinsics[:, 0, :] *=256
            intrinsics[:, 1, :] *=256
            vis_points_images_cameras(w2cs, intrinsics, imgs, frustum_size=0.5, filename=data_dir.split('/')[-2] + 'camemra_ori.html')

        data = {
                'images': imgs,
                'w2cs': w2cs,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'filename': data_dir.split('/')[-2]
        }
        return data
    
    def process_img(self, img):
        img = img.convert("RGB")
        return self.image_transforms(img)