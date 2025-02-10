import torch
import time


def get_sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
