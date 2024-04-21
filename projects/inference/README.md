## How to run

**config file**

The config files for four scenes: carnation, aloacasia, hat, telephone is in configs/ folder. Please check the path for `dataset_dir` and `model_list` is correct after you download all the models. 

**inference.py** 

Please follow `run.sh` for common args. 

If you encounter OOM error, it's very likely due to the Kmeans downsampling operations. See line ~260 of `inference.py`:

``` python
# WARNING: this is a GPU implementation, and will be OOM if the number of points is large
# you might want to use a CPU implementation if the number of points is large
# For CPU implementation: uncomment the following lines
# from local_utils import downsample_with_kmeans
# sim_xyzs = downsample_with_kmeans(sim_xyzs.detach().cpu().numpy(), num_cluster)
# sim_xyzs = torch.from_numpy(sim_xyzs).float().to(device)
sim_xyzs = downsample_with_kmeans_gpu(sim_xyzs, num_cluster)
```

