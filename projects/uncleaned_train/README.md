This folder contains the original uncleaned training code. This folder can be viewed as an independent folder, it did not use code in physdreamer/ and projects/inference

`exp_motion/train` contains code for velocity and material training. 

Velocity train and material train is slightly different:
1. How many frames is used for training.
2. How many frames the backprop needs to be passed. 
3. Velocity train typically use smaller spatial resolution(grid_size) and temporal resolution(num of substeps). 

Two major difference for this code with the inference code is that:
1. All the helper functions here are all installed in a folder called "motionrep". The inference code uses "physdreamer". the physdreamer/ and motionrep/ folder should share most of the code
2. The config.yaml file has different contents and format
