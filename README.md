# diffusion_distillation

TL;DR: 

Simply run
```bash
bash run.sh
```

---

## Env

```bash
conda env create -f dd.yml
```

## Train

1. Adjusting training profiles: `diffusion_distillation/config/mnist_base.py`
2. Run: `diffusion_distillation.ipynb`, The output checkpoint will be saved in `/tmp/flax_ckpt/checkpoint/` directory.
3. Copy the checkpoint to a safe folder, as the /tmp folder will be case on system reboot. 

## Sampling

1. Adjusting sampling profiles: `diffusion_distillation/config/mnist_distill.py`
2. Run the following code to generate images:
```bash
python sample_origin.py --num_imgs 1024 --batchsize 64 --startbatch 0 --db_path data/mnist_origin_debug --ckpt_path /path/to/ckpt
```

- `db_path` means the output dataset, which will be used in [DSNO-pytorch](https://github.com/JavaZeroo/DSNO-pytorch)
- `num_imgs` means the num of images to be generated