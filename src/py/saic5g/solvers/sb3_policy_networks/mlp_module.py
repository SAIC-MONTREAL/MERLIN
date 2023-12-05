from torch import nn


def create_model(input_dim, output_dim, net_arch, activation_fn, output_activation, with_bias):
    # similar to create_mlp in stable_baselines3/common/torch_layers.py
    # but with flexibility in the last activation 
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    # changed here:
    modules.append(output_activation())
    return modules

