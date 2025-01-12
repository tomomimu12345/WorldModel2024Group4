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

    # F[t][n] @ covariance[n] @ F[t][n]^T = FCF^T(t, n, 3, 3)
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

    # 元のデータ型を取得
    original_dtype = F.dtype

    # データ型が float16 の場合は一時的に float32 に変換
    if original_dtype == torch.float16:
        F_matrices = F_matrices.float()

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

    # float16 に戻す（必要な場合のみ）
    if original_dtype == torch.float16:
        R = R.half()

    # (time, nnode, 3, 3) -> (time, nnode, 9)
    R_flat = R.view(F.shape[0], F.shape[1], 9)

    return R_flat

def repeat_to_match_length(tensor, target_length):
    """
    繰り返してテンソルの第一次元を target_length に合わせる。

    Args:
        tensor (torch.Tensor): (time, nnode, feature) の形状のテンソル
        target_length (int): 必要な長さ

    Returns:
        torch.Tensor: 第一次元が target_length に調整されたテンソル
    """
    current_length = tensor.shape[0]

    if current_length >= target_length:
        return tensor[:target_length]

    repeat_factor = (target_length + current_length - 1) // current_length  # 繰り返し回数を計算
    repeated_tensor = tensor.repeat((repeat_factor, 1, 1))  # テンソルを繰り返し
    return repeated_tensor[:target_length]  # 必要な長さだけ切り取る