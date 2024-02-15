# Free3D

<p align="center">
  [<a href=""><strong>arXiv</strong></a>]
  [<a href="https://chuanxiaz.com/free3d/"><strong>Project</strong></a>]
<!--   [<a href=""><strong>Video</strong></a>] -->
  [<a href="#citation"><strong>BibTeX</strong></a>]
</p>

## Teaser example

https://github.com/lyndonzheng/Free3D/assets/8929977/d4888ad6-1a1d-41ee-bc26-35b394a4dfd7

This repository implements the training and testing tools for **Free3D** by [Chuanxia Zheng](http://www.chuanxiaz.com) and [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/) in [VGG](https://www.robots.ox.ac.uk/~vgg/) at the University of Oxford. Given a single-view image, the proposed **Free3D** synthesizes correct novel views without the need of an explicit 3D representation.

## Usage
### Installation
```bash
# create the environment
conda create --name free3d python=3.9
conda activate free3d
# install the pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# other dependencies
pip install -r requirements.txt
```

### Datasets

- [Objaverse](https://objaverse.allenai.org/): For training / evaluating on Objaverse (7,729 instances for testing), please download the rendered dataset from [zero-1-to-3](https://github.com/cvlab-columbia/zero123). The original command they provided is:
  ```
  wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
  ```
  Unzip the data file and change ```root_dir``` in ```configs/objaverse.yaml```.
- [OmniObject3D](https://omniobject3d.github.io/): For evaluating on OmniObject3d (5,275 instances), please refer to [OmniObject3D Github](https://github.com/omniobject3d/OmniObject3D/tree/main), and change ```root_dir``` in ```configs/omniobject3d```. Since we do not train the model on this dataset, we directly evaluate on the training set.
- [GSO](https://app.gazebosim.org/miki/fuel/collections/Scanned%20Objects%20by%20Google%20Research): For evaluating on Google Scanned Objects (GSO, 1,030 instances), please download the whole 3D models, and use the rendered code from [zero-1-to-3](https://github.com/cvlab-columbia/zero123) to get 25 views for each scene. Then, change ```root_dir``` in ```configs/googlescan.yaml``` to the corresponding location. Our rendered files are available on [Google Drive](https://drive.google.com/file/d/1tV-qpiD5e-GzrjW5dQpTRviZa4YV326b/view?usp=drive_link).

### Inference

- batch testing for quantitative results
  ```
  python batch_test.py \
  --resume [model directory path] \
  --config [configs/*.yaml] \
  --save_path [save directory path] \
  ```
- single image testing for qualitative results
  ```
  # for real examples, please download the segment anything checkpoint
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  # run the single image test command
  python test.py \
  --resume [model directory path] \
  --sam_path [sam checkpoint path] \
  --img_path [image path] \
  --gen_type ['image' or 'video'] \
  --save_path [save directory path]
  ```
- the general metrics are evaluated with:
  ```
  cd evaluations
  python evaluation.py --gt_path [ground truth images path] --g_path [generated NVS images path]
  ```

### Training
- The Ray Conditioning Normalization (RCN) to enhance the pose accuracy is trained with the following command:
  ```
  # download image-conditional stable diffusion checkpoint released by lambda labs
  # this training takes around 9 days on 4x a6000 (48G)
  wget https://cv.cs.columbia.edu/zero123/assets/sd-image-conditioned-v2.ckpt
  # or download checkpoint released by zero-1-t-3
  # this training takes around 2 days on 4x 60000 (48G)
  wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt
  # change the finetune_from in train.sh, and run the command
  sh train.sh
  ```
- The pseudo 3D attention to smooth the consistency is trained with the same command (1 day on 4x A6000), but with different parameters:
  ```
  # modify the configs/objaverse.yaml as follows
  views: 4
  use_3d_transformer: True
  # modify the finetune_from in train.sh to you first stage model
  finetune_from [RCN trained model]
  ```

### Pretrained models
- RCN w/o pseudo 3D attention model is available at [huggingface](https://huggingface.co/lyndonzheng/Free3D/tree/main).
- All models will be released after ECCV DDL.

## Related work
- [Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) fine-tunes image-to-video diffusion model for multi-view generations.
- [Efficient-3DiM](https://arxiv.org/pdf/2310.03015.pdf) fine-tunes the stable diffusion model with a stronger vision transformer [DINO v2](https://dinov2.metademolab.com/).
- [Consistent-1-to-3](https://jianglongye.com/consistent123/) applies the epipolar-attention to extract coarse results for the diffusion model.
- [One-2-3-45](https://one-2-3-45.github.io/) and [One-2-3-45++](https://sudo-ai-3d.github.io/One2345plus_page/) train additional 3D network using the outputs of 2D generator.
- [MVDream](https://mv-dream.github.io/), [Consistent123](https://consistent-123.github.io/index.html) and [Wonder3D](https://www.xxlong.site/Wonder3D/) also train multi-view diffusion models, yet still require post-processing for video rendering.
- [SyncDreamer](https://liuyuan-pal.github.io/SyncDre32qwDeamer/) and [ConsistNet](https://jiayuyang.github.io/Consist_Net/) employ 3D representation into the latent diffusion model.

## Citation

If you find our code helpful, please cite our paper:

```
@article{zheng2023free3D,
      author    = {Zheng, Chuanxia and Vedaldi, Andrea},
      title     = {Free3D: Consistent Novel View Synthesis without 3D Representation},
      journal   = {arXiv},
      year      = {2023},
```

## Acknowledgements
Many thanks to [Stanislaw Szymanowicz](https://dblp.org/pid/295/8991.html), [Edgar Sucar](https://edgarsucar.github.io/), and [Luke Melas-Kyriazi](https://lukemelas.github.io/) of VGG for insightful discussions and [Ruining Li](https://ruiningli.com/), [Eldar Insafutdinov](https://eldar.insafutdinov.com/), and [Yash Bhalgat](https://yashbhalgat.github.io/) of VGG for their helpful feedback. We would also like to thank the authors of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) and [Objaverse-XL](https://github.com/allenai/objaverse-xl) for their helpful discussions.
