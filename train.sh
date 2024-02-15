python main.py \
    -t \
    --base configs/objaverse.yaml \
    --logdir /work/cxzheng/diff3d/test/logs \
    --name test \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from /work/cxzheng/code/zero123_old/zero123/105000.ckpt
    # --finetune_from /work/cxzheng/code/zero123_old/zero123/sd-image-conditioned-v2.ckpt