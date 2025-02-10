

import numpy as np

def latent_sds(input_x, schduler, unet, t_range=[0.02, 0.98]):

    # t_range_annel: [0.02, 0.98] => [0.50, 0.98]
    # input_x: # [T, 4, H, W] 



    sigma = schduler.sample_sigma(t_range) # scalar

    noise = randn_like(input_x)

    noised_latent = input_x + sigma * noise

    c, uc = None 
    # x0 prediction. 
    denoised_latent_c, denoised_latent_uc = unet(noised_latent, c, uc)

    w = [1.0, 2.0, 3.0]
    denoised_latent = denoised_latent_uc + w * (denoised_latent_c - denoised_latent_uc)

    sds_grad = (input_x - denoised_latent) / sigma

    loss_sds = MSE(input_x - (input_x - sds_grad).detach())
