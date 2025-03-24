import torch
from bingham.gram_schmidt import gram_schmidt
from torch import nn


def load_norm_const_model(device: torch.device):
    path = "rl_solver/bingham/precomputed/norm_constant_model_-500_0_200.model"
    norm_const_model = nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1))
    norm_const_model.to(device)
    norm_const_model.load_state_dict(torch.load(path, map_location=device))
    norm_const_model.train(False)
    for p in norm_const_model.parameters():
        p.requires_grad = False
    return norm_const_model


def load_b_model(device: torch.device):
    path = "rl_solver/bingham/precomputed/norm_constant_model_-500_0_200.model"
    bmodel = nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1))
    bmodel.to(device)
    bmodel.load_state_dict(torch.load(path, map_location=device))
    bmodel.train(False)
    for p in bmodel.parameters():
        p.requires_grad = False
    return bmodel


def vec_to_bingham_z_many(z):
    z1 = torch.minimum(z[:, :1], z[:, 1:2])
    z2 = torch.minimum(z1, z[:, 2:3])
    z = -torch.exp(torch.cat([z[:, :1], z1, z2], 1))
    return z


def vec19_to_m_z(output):
    bs = output.shape[0]
    bd_z = vec_to_bingham_z_many(output[:, :3])
    bd_m = gram_schmidt(output[:, 3:].reshape(bs, 4, 4))
    return bd_m, bd_z
