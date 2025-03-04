#%%
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,Lambda,ToTensor

import torch
import torch.nn as nn
from torch.nn.utils.prune import l1_unstructured, random_unstructured

import math

# %%
class MNISTDataset(Dataset):
    """MNIST dataset
    
    Feature images are automatically flattened.
    
    Parameters
    ----------
    root    : str
        Directory where the data is located (or downloaded to).
    
    train   : bool
        If True the training set is returned (60_000 samples). Otherwise
        the validation set is returned (10_000 samples).
    
    Attributes
    ----------
    tv_dataset  : MNIST
        Instance of the torchvision MNIST dataset class.
    """
    def __init__(self, root, train=True, download=True):
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: x.ravel()),
        ])
        self.tv_dataset = MNIST(
            root,
            train=train,
            download=download,
            transform=transform
        )
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.tv_dataset)

    def __getitem__(self, idx):
        """Get a selected sample.
        
        Parameters
        ----------
        idx : int
            Index of the sample to get.
        
        Return
        ------
        x   : torch.Tensor
            Flatten feature tensor of shape `(784,)`.
            
        y   : torch.Tensor
            Scalar representing the ground truth label. Number between 0 and 9.
        """
        return self.tv_dataset[idx]


# %%
class MLP(nn.Module):
    """Multilayer perceptron"""
    def __init__(self, n_features, hidden_layer_size, n_targets):
        super().__init__()
        
        layer_sizes = (n_features,) + hidden_layer_size + (n_targets,)
        layer_list = []
        
        for i in range(len(layer_sizes)-1):
            layer_list.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.module_list = nn.ModuleList(layer_list)
    
    def forward(self, x):
        n_layers = len(self.module_list)
        
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            if i < n_layers-1:
                x = nn.functional.relu(x)
        
        return x

#%%
def prune_linear(linear, prune_ratio=0.3, method="l1"):
    if method == "l1":
        prune_func = l1_unstructured
    elif method == "random":
        prune_func = random_unstructured
    else:
        raise ValueError("pruning method must l1 or random.")
    
    prune_func(linear, "weight", prune_ratio)
    prune_func(linear, "bias", prune_ratio)

def prune_mlp(mlp, prune_ratio=0.3, method="l1"):
    if isinstance(prune_ratio, float):
        prune_ratios = [prune_ratio] * len(mlp.module_list)
    elif isinstance(prune_ratio, list):
        if len(prune_ratio) != len(mlp.module_list):
            raise ValueError("Incompatible number of prune ratios provided")
        prune_ratios = prune_ratio
    else:
        raise TypeError
    
    for prune_ratio, linear in zip(prune_ratios, mlp.module_list):
        prune_linear(linear,prune_ratio=prune_ratio, method=method)

def check_pruned_linear(linear):
    params = {param_name for param_name,_ in linear.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}
    return params == expected_params

def reinit_linear(linear):
    is_pruned = check_pruned_linear(linear)
    
    # get parameters of interest
    if is_pruned:
        weight = linear.weight_orig
        bias = linear.bias_orig
    else:
        weight = linear.weight
        bias = linear.bias
    
    # initialize weights
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    # initialize bias
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

def reinit_mpl(mpl):
    for linear in mpl.module_list:
        reinit_linear(linear)

def copy_weights_linear(linear_unpruned, linear_pruned):
    assert check_pruned_linear(linear_pruned)
    assert not check_pruned_linear(linear_unpruned)
    
    with torch.no_grad():
        linear_pruned.weight_orig.copy_(linear_unpruned.weight)
        linear_pruned.bias_orig.copy_(linear_unpruned.bias)

def copy_weight_mlp(mlp_unpruned, mlp_pruned):
    