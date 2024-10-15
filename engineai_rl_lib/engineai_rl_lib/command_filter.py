import torch


def convert_to_visible_commands(commands, mins=[0.5, 0.25, 0.5]):
    for i, threshold in enumerate(mins):
        sign = torch.where(
            torch.sign(commands[:, i]) == 0, 1, torch.sign(commands[:, i])
        )
        commands[:, i] = sign * torch.max(
            torch.abs(commands[:, i]), torch.ones_like(commands[:, i]) * threshold
        )
