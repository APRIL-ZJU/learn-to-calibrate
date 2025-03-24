"""
Credit for this file goes to:
Gilitschenski, Igor, et al. "Deep orientation uncertainty learning based
on a bingham loss." International Conference on Learning Representations. 2019.
"""

import torch


def _gram_schmidt(input_mat, reverse=False, modified=False):
    """ Carries out the Gram-Schmidt orthogonalization of a matrix.

    Arguments:
        input_mat (torch.Tensor): A quadratic matrix that will be turned into an
            orthogonal matrix.
        reverse (bool): Starts gram Schmidt method beginning from the last
            column if set to True.
        modified (bool): Uses modified Gram-Schmidt as described.
    """
    mat_size = input_mat.shape[0]
    Q = torch.zeros(mat_size, mat_size,
                    device=input_mat.device, dtype=input_mat.dtype)

    if modified:
        if reverse:
            outer_iterator = range(mat_size - 1, -1, -1)
            def inner_iterator(k): return range(k, -1, -1)
        else:
            outer_iterator = range(mat_size)
            def inner_iterator(k): return range(k+1, mat_size)

        # This implementation mostly follows the description from
        # https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html
        # The more complex form is due to pytorch not allowing for inplace
        # operations of variables needed for gradient computation.
        v = input_mat
        for j in outer_iterator:
            Q[:, j] = v[:, j] / torch.norm(v[:, j])

            v_old = v
            v = torch.zeros(mat_size, mat_size,
                            device=input_mat.device, dtype=input_mat.dtype)

            for i in inner_iterator(j):
                v[:, i] = v_old[:, i] \
                          - (torch.dot(Q[:, j].clone(), v_old[:, i])
                             * Q[:, j].clone())

    elif not modified:
        if reverse:
            outer_iterator = range(mat_size - 1, -1, -1)
            def inner_iterator(k): return range(mat_size - 1, k, -1)
        else:
            outer_iterator = range(mat_size)
            def inner_iterator(k): return range(k)

        for j in outer_iterator:
            v = input_mat[:, j]
            for i in inner_iterator(j):
                p = torch.dot(Q[:, i].clone(), v) * Q[:, i].clone()
                v = v - p

            Q[:, j] = v / torch.norm(v)

    return Q


def gram_schmidt(input_mat, reverse=False, modified=False):
    """ Carries out the Gram-Schmidt orthogonalization of a matrix on an
        entire batch.

    Arguments:
        input_mat (torch.Tensor): A tensor containing quadratic matrices each of
            which will be orthogonalized of shape (batch_size, m, m).
        reverse (bool): Starts gram Schmidt method beginning from the last
            column if set to True.
        modified (bool): Uses modified Gram-Schmidt as described.

    Returns:
        Q (torch.Tensor): A batch of orthogonal matrices of same shape as
            input_mat.
    """
    batch_size = input_mat.shape[0]
    mat_size = input_mat.shape[1]
    Q = torch.zeros(batch_size, mat_size, mat_size,
                    device=input_mat.device, dtype=input_mat.dtype)

    if modified:
        for i in range(input_mat.shape[0]):
            q = _gram_schmidt(input_mat[i], reverse, modified)
            Q[i] = q
    elif not modified:
        if reverse:
            raise NotImplementedError
        else:
            outer_iterator = range(mat_size)
            def inner_iterator(k): return range(k)

        for j in outer_iterator:
            v = input_mat[:, :, j].view(batch_size, mat_size, 1)
            for i in inner_iterator(j):
                q_squeezed = Q[:, :, i].view(batch_size, 1, mat_size).clone()
                dot_products = torch.bmm(q_squeezed, v)
                p = dot_products.repeat((1, mat_size, 1)) \
                    * Q[:, :, i].unsqueeze(2).clone()
                v = v - p

            Q[:, :, j] = (v.squeeze()) / torch.max(torch.norm(v, dim=1), torch.Tensor([1e-9]).cuda()).repeat(1, mat_size)

    return Q
