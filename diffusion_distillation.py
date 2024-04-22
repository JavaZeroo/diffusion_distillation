import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import requests
import functools
import jax
from jax import config
import jax.numpy as jnp
import flax
from matplotlib import pyplot as plt
import numpy as onp
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import diffusion_distillation
from flax.training import checkpoints as flaxcheckpoints



# %% [markdown]
# ## Train a new diffusion model

# %%
# jax.devices()
# jax.config.FLAGS.jax_xla_backend = "gpu_driver"
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)
jax.devices()


# %%
# create model
config = diffusion_distillation.config.mnist_base.get_config()
model = diffusion_distillation.model.Model(config)

# %%
print(model.model)

# %%
print(type(model.make_optimizer_def()))
# print(type(model.make_optimizer_def().init(None)))

# %%
# init params 
state = jax.device_get(model.make_init_state())
state = flax.jax_utils.replicate(state)

# %%
type(model.make_optimizer_def())

# %%
# type(state.optimizer[0]), type(state.optimizer[1])

# %%
model.make_init_state()

# %%
# JIT compile training step
train_step = functools.partial(model.step_fn, jax.random.PRNGKey(0), True)
train_step = functools.partial(jax.lax.scan, train_step)  # for substeps
train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

# %%
# build input pipeline
total_bs = config.train.batch_size
device_bs = total_bs // jax.device_count()
train_ds = model.dataset.get_shuffled_repeated_dataset(
    split='train',
    batch_shape=(
        jax.local_device_count(),  # for pmap
        config.train.substeps,  # for lax.scan over multiple substeps
        device_bs,  # batch size per device
    ),
    local_rng=jax.random.PRNGKey(0),
    augment=True)
train_iter = diffusion_distillation.utils.numpy_iter(train_ds)

# %%
config.model.mean_type

# %%
# run training
for step in range(1000):
  batch = next(train_iter)
  state, metrics = train_step(state, batch)
  print(flaxcheckpoints.save_checkpoint('/tmp/flax_ckpt/checkpoint/', state, step, overwrite=True))
  metrics = jax.device_get(flax.jax_utils.unreplicate(metrics))
  metrics = jax.tree_map(lambda x: float(x.mean(axis=0)), metrics)
  print(metrics)

# %%
# get all attr
#print(dir(state))
#for k, v in dict(state.ema_params).items():
#    print(k, type(v))
#print()
print(state.num_sample_steps)
print(model.make_init_state().num_sample_steps)
#print(state.optimizer)
#print(state.replace)
#print(state.step)