# Free3D

<p align="center">
  [<a href=""><strong>arXiv</strong></a>]
  [<a href="https://chuanxiaz.com/free3d/"><strong>Project</strong></a>]
  [<a href="https://youtu.be/7CdYuZ7D1DY"><strong>Video</strong></a>]
  [<a href="#citation"><strong>BibTeX</strong></a>]
</p>

## Teaser example

https://github.com/lyndonzheng/Free3D/assets/8929977/d4888ad6-1a1d-41ee-bc26-35b394a4dfd7

This repository implements the training and testing tools for **Free3D** by [Chuanxia Zheng](http://www.chuanxiaz.com) and [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/) in [VGG](https://www.robots.ox.ac.uk/~vgg/) at the University of Oxford. Given a single-view image, the proposed **Free3D** synthesizes consistent novel views without the need of an explicit 3D representation.

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
