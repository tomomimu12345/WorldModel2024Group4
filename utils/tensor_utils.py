import torch
def expand_covariance(cov_flat):
    """
    Expand a flattened covariance matrix of shape (time, nnode, 6)
    into a symmetric 3x3 matrix of shape (time, nnode, 3, 3).
    
    Args:
        cov_flat (torch.Tensor): Tensor of shape (time, nnode, 6),
                                 where each row is [c00, c01, c02, c11, c12, c22].
    
    Returns:
        torch.Tensor: Expanded tensor of shape (time, nnode, 3, 3).
    """
    time, nnode = cov_flat.shape[:2]
    cov_matrix = torch.zeros((time, nnode, 3, 3), device=cov_flat.device, dtype=cov_flat.dtype)

    # Fill the symmetric 3x3 matrix
    cov_matrix[..., 0, 0] = cov_flat[..., 0]  # c00
    cov_matrix[..., 0, 1] = cov_flat[..., 1]  # c01
    cov_matrix[..., 0, 2] = cov_flat[..., 2]  # c02
    cov_matrix[..., 1, 0] = cov_flat[..., 1]  # c01 (symmetric)
    cov_matrix[..., 1, 1] = cov_flat[..., 3]  # c11
    cov_matrix[..., 1, 2] = cov_flat[..., 4]  # c12
    cov_matrix[..., 2, 0] = cov_flat[..., 2]  # c02 (symmetric)
    cov_matrix[..., 2, 1] = cov_flat[..., 4]  # c12 (symmetric)
    cov_matrix[..., 2, 2] = cov_flat[..., 5]  # c22

    return cov_matrix


def flatten_covariance(cov_matrix):
    """
    Flatten a symmetric 3x3 covariance matrix of shape (time, nnode, 3, 3)
    into a compact 6-element representation of shape (time, nnode, 6).
    
    Args:
        cov_matrix (torch.Tensor): Tensor of shape (time, nnode, 3, 3).
    
    Returns:
        torch.Tensor: Flattened tensor of shape (time, nnode, 6), where each row is [c00, c01, c02, c11, c12, c22].
    """
    time, nnode = cov_matrix.shape[:2]
    cov_flat = torch.zeros((time, nnode, 6), device=cov_matrix.device, dtype=cov_matrix.dtype)

    # Extract the symmetric matrix elements
    cov_flat[..., 0] = cov_matrix[..., 0, 0]  # c00
    cov_flat[..., 1] = cov_matrix[..., 0, 1]  # c01
    cov_flat[..., 2] = cov_matrix[..., 0, 2]  # c02
    cov_flat[..., 3] = cov_matrix[..., 1, 1]  # c11
    cov_flat[..., 4] = cov_matrix[..., 1, 2]  # c12
    cov_flat[..., 5] = cov_matrix[..., 2, 2]  # c22

    return cov_flat



def compute_transformed_covariance(F, covariance):
    """
    Args:
        F: Tensor of shape(time, nnode, 9)
        covariance: Tensor of shape (nnode, 3, 3)
    Return:
        torch.Tensor: (time, nnode, 3, 3)
    """
    # (time, nnode, 9) -> (time, nnode, 3, 3)
    F_matrices = F.view(F.shape[0], F.shape[1], 3, 3)

    # F[t][n] @ covariance[n] @ F[t][n]^T
    result = torch.einsum(
        "tnij,njk,tnlk->tnil", F_matrices, covariance, F_matrices
    )
    return result

def compute_R_from_F_pytorch(F):
    """
    F: torch.Tensor of shape (time, nnode, 9), where time is the number of time steps
       and nnode is the number of particles.
    Returns:
        R_flat: torch.Tensor of shape (time, nnode, 9), the flattened rotation matrices for each particle at each time step.
    """
    # (time, nnode, 9) -> (time, nnode, 3, 3)
    F_matrices = F.view(F.shape[0], F.shape[1], 3, 3)

    # SVD分解
    U, S, Vt = torch.linalg.svd(F_matrices)

    # UとVtの行列式をチェックし、反射成分を修正
    det_U = torch.det(U)
    det_Vt = torch.det(Vt)

    # 修正が必要な場合は最後の列を反転
    U[..., :, 2] *= torch.where(det_U < 0, -1.0, 1.0).unsqueeze(-1)
    Vt[..., 2, :] *= torch.where(det_Vt < 0, -1.0, 1.0).unsqueeze(-1)

    # 回転行列の計算
    R = torch.matmul(U, Vt)
    
    # (time, nnode, 3, 3) -> (time, nnode, 9)
    R_flat = R.view(F.shape[0], F.shape[1], 9)
    
    return R_flat