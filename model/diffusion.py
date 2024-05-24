"""
This code is based from
https://github.com/taki0112/diffusion-pytorch
"""

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from model.modules import linear_beta_schedule, cosine_beta_schedule
# from modules import linear_beta_schedule, cosine_beta_schedule
import logging
import torch.nn.functional as F

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, objective='ddpm', schedule='linear', device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.objective = objective

        self.beta = self.prepare_noise_schedule(schedule, beta_start, beta_end)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule, beta_start, beta_end):
        if schedule == 'linear':
            return linear_beta_schedule(self.noise_steps, beta_start, beta_end)
        else:
            return cosine_beta_schedule(self.noise_steps)

    def noise_samples(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None].to(x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None].to(x.device)
        z = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * z, z

    def sample_timesteps(self, n):
        t = torch.randint(low=1, high=self.noise_steps, size=(n,))
        return t

    def tensor_to_image(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x
    
    def sample(self, predicted_token):
        # reverse process
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(predicted_token.size()[0], dtype=torch.long) * i).to(self.device)

                alpha = self.alpha[t][:, None, None, None].to(predicted_token.device)
                beta = self.beta[t][:, None, None, None].to(predicted_token.device)
                alpha_hat = self.alpha_hat[t][:, None, None, None].to(predicted_token.device)
                alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None].to(predicted_token.device)
                beta_tilde = beta * (1 - alpha_hat_prev) / (1 - alpha_hat) # similar to beta

                noise = torch.randn_like(predicted_token)

                predict_x0 = 0
                direction_point = 1 / torch.sqrt(alpha) * predicted_token
                random_noise = torch.sqrt(beta_tilde) * noise

                x = predict_x0 + direction_point + random_noise

        return torch.clamp(x, -1.0, 1.0)

    
