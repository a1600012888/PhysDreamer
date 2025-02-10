import torch.distributed as dist


def print_if_zero_rank(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)
