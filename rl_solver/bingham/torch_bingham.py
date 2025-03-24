import torch
from torch.distributions import MultivariateNormal, Uniform

from bingham.utils import load_norm_const_model, load_b_model


class BinghamDistribution(object):

    def __init__(self,
                 device: torch.device,
                 orthogonalization="gram_schmidt"):
        self.orthogonalization = orthogonalization
        self._norm_const_model = load_norm_const_model(device)
        self._b_model = load_b_model(device)

    def rsample(self, M, Z):
        device = M.get_device()
        if device == -1:
            device = 'cpu'
        n, _, dim = M.shape
        z_padded = torch.cat(
            (Z, torch.zeros((Z.shape[0], 1), device=device, dtype=M.dtype)),
            dim=1)
        z_as_matrices = torch.diag_embed(z_padded)

        a = -torch.bmm(M, torch.bmm(z_as_matrices, M.transpose(1, 2)))
        b = self._b_model(Z)[:, 0]
        b = torch.max(b, torch.zeros_like(b) + 1e-3)
        self.b_v = b
        omega = torch.eye(dim, device=device).unsqueeze(
            0) + 2. * a / b.unsqueeze(-1).unsqueeze(-1)
        mbstar = torch.exp(-(dim - b) / 2.) * (dim / b) ** (dim / 2.)

        def fb_likelihood(x):
            # fold sample axis into batch axis
            x = x.view(n * SAMPLES, -1)
            aa = a.unsqueeze(1).repeat(1, SAMPLES, 1, 1).view(n * SAMPLES, 4, 4)
            y = torch.exp(
                torch.bmm(-x.unsqueeze(1), torch.bmm(aa, x.unsqueeze(-1))))
            return y.view(n, SAMPLES, 1)

        def acg_likelihood(x):
            x = x.view(n * SAMPLES, -1)
            oo = omega.unsqueeze(1).repeat(1, SAMPLES, 1, 1).view(n * SAMPLES,
                                                                  4, 4)
            y = (torch.bmm(x.unsqueeze(1), torch.bmm(oo, x.unsqueeze(-1))) ** (
                        -dim / 2.))
            return y.view(n, SAMPLES, 1)

        SAMPLES = 10
        u = Uniform(torch.zeros((n, SAMPLES, 1), device=device),
                    torch.ones((n, SAMPLES, 1), device=device))
        m = MultivariateNormal(torch.zeros(
            (n, SAMPLES, dim), device=device, dtype=omega.dtype),
            torch.inverse(omega).unsqueeze(1).repeat(1, SAMPLES, 1, 1))
        candidates = m.rsample()  # (B, samples, dim)
        candidates = candidates / candidates.norm(dim=2, keepdim=True)
        w = u.rsample()
        mask = (w < fb_likelihood(candidates) / (
                    mbstar.unsqueeze(-1).unsqueeze(-1) * acg_likelihood(
                candidates))).float().repeat(1, 1, 4)
        samples = candidates.gather(1, mask.sort(dim=1, descending=True)[1])[:,
                  0]

        return samples

    def log_probs(self, x, M, Z):
        device = x.get_device()
        if device == -1:
            device = 'cpu'

        z_padded = torch.cat(
            (Z, torch.zeros((Z.shape[0], 1), device=device, dtype=M.dtype)),
            dim=1)
        z_as_matrices = torch.diag_embed(z_padded)

        norm_const = self._norm_const_model(Z)[:, 0]
        norm_const = torch.maximum(norm_const,
                                   torch.zeros_like(norm_const) + 1e-6)
        likelihoods = (torch.bmm(torch.bmm(torch.bmm(torch.bmm(
            x.unsqueeze(1), M), z_as_matrices), M.transpose(1, 2)),
            x.unsqueeze(2))).squeeze() - torch.log(norm_const)
        return likelihoods.reshape(likelihoods.shape[0],1)

    def entropy(self, Z):
        with torch.enable_grad():
            Z_ = Z.clone().detach()
            Z_.requires_grad = True
            norm_const = self._norm_const_model(Z_)[:, 0]
            norm_const = torch.maximum(norm_const,
                                       torch.zeros_like(norm_const) + 1e-6)
            dz, = torch.autograd.grad(outputs=norm_const, inputs=Z_,
                                      grad_outputs=torch.ones_like(norm_const))
            dz = dz.detach()
        norm_const = self._norm_const_model(Z_)[:, 0]
        norm_const = torch.maximum(norm_const,
                                   torch.zeros_like(norm_const) + 1e-6)
        entropy = torch.log(norm_const) - (
                    Z * (dz / norm_const.unsqueeze(1))).sum(1)
        return entropy
