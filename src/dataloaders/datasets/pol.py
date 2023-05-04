import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def memory_pol(t, rho_sum=1, power=1.2,):

    return (power-1)/(t+1)**(power) / rho_sum


def torch_pol_data(L, configs, batch_shape=()):

    inputs = np.random.normal(size=(
        configs['size'],
        configs['path_len'],
        configs['input_dim'],
    ))
    # inputs = np.cumsum(inputs, axis=1) # Make it Gaussian
    
    outputs = []
    for t in range(configs['path_len']):
        output = 0
        for s in range(t + 1):
            output += inputs[:, t - s, :] * (np.power(np.abs(inputs[:, t - s, :]), configs['power'] - 1)) * (configs['rho'](s * configs['dt'])) 
        outputs.append(output)
    # return inputs, np.asarray(outputs).transpose(1, 0, 2)
    return inputs, output


def pol_static_dataset(L, samples):
    
    generator_config = {
        'size': samples,
        'path_len': L,
        'input_dim': 1,
        'rho': memory_pol,
        'dt': 0.1,
        'power': 1.0,
    }

    all_x, all_y = torch_pol_data(L, generator_config, batch_shape=(samples,))
    print("Constructing polynomial memory dataset", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds
