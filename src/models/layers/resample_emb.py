
import torch
import torch.nn.functional as F
import math

def resample_patchemb(old: torch.Tensor, new_patch_len: int):

    assert old.dim() == 2, "输入张量应为2D (d_model, patch_size)"
    if old.size(1) == new_patch_len:
        return old

    old = old.T
    old_shape = old.size(0)
    factor = new_patch_len/old_shape
    
    # 定义辅助函数：批量resize
    def resize(x_tensor, new_shape):
        return F.interpolate(x_tensor.unsqueeze(0), size=new_shape, mode='linear').squeeze(0)

    # 构造缩放矩阵
    basis_vectors = torch.eye(old_shape, dtype=torch.float32, device=old.device)
    resize_mat = resize(basis_vectors, new_patch_len).T
    # 计算伪逆
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)
    
    # z_inverse = z @ resize_mat_pinv
    # z_inverse_var = z_inverse.var(dim=-1).mean(dim=1).mean()
    # z_var = z.var(dim=-1).mean(dim=1).mean()
    # z_interpolate = z_inverse @ resize_mat.T
    # z_interpolate_var = z_interpolate.var(dim=-1).mean(dim=1).mean()

    # print(z_inverse_var)
    # print(z_var)
    # print(z_interpolate_var/z_inverse_var)


    # 直接矩阵操作完成重采样
    resampled_kernels = resize_mat_pinv @ old * math.sqrt(factor)

    return resampled_kernels.T



# def resample_patchemb(old, new_patch_len):
#     new_patch_size = new_patch_len
#     """Resample the weights of the patch embedding kernel to target patch size."""
#     old = old.T

#     # 获取原始大小
#     patch_size, d_model = old.shape
#     factor = new_patch_len/patch_size
    
#     if patch_size == new_patch_size:
#         return old.T

#     # 将原始权重张量扩展为四维，以便可以使用 F.interpolate
#     old = old.permute(1, 0).unsqueeze(1)

#     # 使用 F.interpolate 调整 patch_size 维度
#     resampled = F.interpolate(old, size=new_patch_size, mode='linear')/math.sqrt(factor)

#     # 移除多余的批次和通道维度
#     resampled = resampled.squeeze(1).permute(1, 0)

    
#     return resampled.T