ngc batch run --name "ml-model.exempt-imagenet-mergedb" --preempt RUNONCE --commandline  'cd /GAN-code/meta_model/diffusion_distillation; git config --global --add safe.directory /GAN-code/meta_model/diffusion_distillation; git pull; python3 merge_db.py' --image "nvidian/diff-dist:0.0.1" --ace nv-us-west-2 --instance cpu.x86.tiny --result /results --workspace O7-0rdpyTiqLbdKYtM0Lkw:/GAN-code --port 6006 --port 1234 --port 8888
