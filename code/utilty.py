import os
import random
import numpy as np
from PIL import Image
import torch
from torch import Tensor

USE_GPU = torch.cuda.is_available()
print("USE_GPU: ", USE_GPU)

seed = 31101995
SEED = 31101995

def seed_everything(seed= SEED):
    """Seed every random generator for reproducibility."""
    print("...... .. Setting Seed... ... ...")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backend.cudnn.deterministic = True
    
    
def to_gpu(tendor):
    """Send tensor to GPU if available."""
    return torch.cuda() if USE_GPU else tensor

def T(tensor):
    """Send tensor to GPU if available & convert it to foat Tensor."""
    if not torch.is_tensor(tensor):
        tensor = torch.FloatTensor(tensor)
    else:
        tensor = tensor.type(torch.FloatTensor)
        
    if USE_GPU:
        tensor = to_gpu(tensor)
    return tensor

def to_numpy(tensor):
    """Convert tensor to numpy array."""
    if type(tensor) == np.array or type(tensor) == np.ndarray:
        return np.array(tensor)
    elif type(tensor) == Image.Image:
        return np.array(tensor)
    elif type(tensor) == Tensor:
        return tensor.cpu().detach().numpy()
    else:
        msg = "Unknon datatype."
        raise ValueError(msg)
    
