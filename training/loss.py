# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Consistency Models"

@persistence.persistent_class
class CTLoss:
    def __init__(self, N=2, rho=7, sigma_min=0.002, sigma_max=80., sigma_data=0.5):
        self.N = N
        self.rho = rho
        self.s_min = sigma_min
        self.s_max = sigma_max
        self.s_data = sigma_data

    def __call__(self, net, net_target, images, labels=None, augment_pipe=None):
        rnd_idx = torch.randint(1, self.N, [images.shape[0], 1, 1, 1], device=images.device)
        n_steps = torch.arange(self.N, dtype=torch.float64, device=images.device)
        all_sigma = (self.s_max ** (1 / self.rho) + n_steps / (self.N - 1) * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))) ** self.rho
        # intuitively, using smaller sigma in target net could be helpful
        sigma = all_sigma[rnd_idx - 1]
        sigma_target = all_sigma[rnd_idx]
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        z = torch.randn_like(y)
        n = z * sigma
        n_target = z * sigma_target
        D_yn_target = net_target(y + n_target, sigma_target, labels, augment_labels=augment_labels)
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = (D_yn - D_yn_target) ** 2
        return loss
        

#----------------------------------------------------------------------------
# Test New loss function

@persistence.persistent_class
class TestLoss:
    def __init__(self, N=2, rho=7, data_size=50000, sigma_min=0.002, sigma_max=80., sigma_data=0.5):
        self.N = N
        self.rho = rho
        self.data_size = data_size
        self.s_min = sigma_min
        self.s_max = sigma_max
        self.s_data = sigma_data

    def __call__(self, net, net_target, images, labels=None, augment_pipe=None):
        b_size = images.shape[0]
        scale = np.log((b_size - 1) / (self.data_size - 1))
        rnd_idx = torch.randint(1, self.N, [b_size, 1, 1, 1], device=images.device)
        n_steps = torch.arange(self.N, dtype=torch.float64, device=images.device)
        all_sigma = (self.s_max ** (1 / self.rho) + n_steps / (self.N - 1) * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))) ** self.rho
        sigma = all_sigma[rnd_idx - 1]
        sigma_target = all_sigma[rnd_idx]
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        yn = y + torch.randn_like(y) * sigma
        cdist = torch.cdist(yn.view(b_size, -1), images.view(b_size, -1).double())
        logp = - 2 * cdist / (sigma + sigma_target).view(b_size, 1) ** 2
        logp += torch.diag(torch.ones_like(torch.diag(logp)) * scale)
        # b * 1 * b @ (b * 1 * d - 1 * b * d) = b * 1 * d -> b * d
        grad = (torch.softmax(logp, dim=1).unsqueeze(1) @ (yn.unsqueeze(1) - images.unsqueeze(0).double()).view(b_size, b_size, -1)).squeeze(1)
        yn_target = (yn + grad.view(images.shape) / sigma * (sigma_target - sigma)).detach()
        D_yn_target = net_target(yn_target, sigma_target, labels, augment_labels=augment_labels)
        D_yn = net(yn, sigma, labels, augment_labels=augment_labels)
        loss = (D_yn - D_yn_target) ** 2
        return loss

