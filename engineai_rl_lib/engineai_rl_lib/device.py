import torch


def input_to_device(tensor_input, device):
    if isinstance(tensor_input, torch.Tensor):
        return tensor_input.to(device)
    elif isinstance(tensor_input, dict):
        input_on_device = {}
        for key, item in tensor_input.items():
            input_on_device[key] = input_to_device(item, device)
        return input_on_device
    elif isinstance(tensor_input, list):
        input_on_device = []
        for single_input in tensor_input:
            input_on_device.append(input_to_device(single_input, device))
        return input_on_device
    elif isinstance(tensor_input, tuple):
        input_on_device = []
        for single_input in tensor_input:
            input_on_device.append(input_to_device(single_input, device))
        return tuple(
            single_input_on_device for single_input_on_device in input_on_device
        )
