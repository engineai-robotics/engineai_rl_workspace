from engineai_gym.envs.base.goals.goals import Goals
import torch


class GoalsBiped(Goals):
    def pos_phase(self):
        phase = self.env.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        return torch.cat((sin_pos, cos_pos), dim=1)
