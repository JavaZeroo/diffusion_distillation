# Env

```bash
conda env create -f dd.yml
```

# Train

1. Adjusting training profiles: `diffusion_distillation/config/mnist_base.py`
2. Run: `diffusion_distillation.ipynb`, The output checkpoint will be saved in `/tmp/flax_ckpt/checkpoint/` directory.
3. Copy the checkpoint to a safe folder, as the /tmp folder will be case on system reboot. 