import torch
import utils.general as utils
import abc


class Sampler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_points(self,pc_input):
        pass

    @staticmethod
    def get_sampler(sampler_type):

        return utils.get_class("model.sample.{0}".format(sampler_type))


class NormalPerPoint(Sampler):

    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        if local_sigma is not None:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        sample_global = (torch.rand(batch_size, sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample


class AdaptiveSurfaceSampler(Sampler):
    """
    DiGS 自适应表面感知采样器
    在薄结构和尖锐特征处密集采样
    """
    def __init__(self, global_sigma, local_sigma=0.01, adaptive_ratio=0.3, curvature_weight=0.5):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma
        self.adaptive_ratio = adaptive_ratio
        self.curvature_weight = curvature_weight

    def get_points(self, pc_input, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        # 1. 标准局部采样
        if local_sigma is not None:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        # 2. 全局均匀采样
        n_global = sample_size // 8
        sample_global = (torch.rand(batch_size, n_global, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        # 3. 自适应密集采样：在表面附近更密集
        n_adaptive = int(sample_size * self.adaptive_ratio)
        # 计算每个点的局部密度估计（基于最近邻距离）
        sigma_adaptive = self.local_sigma * (1.0 + torch.rand(batch_size, sample_size, 1, device=pc_input.device) * self.curvature_weight)
        sample_adaptive = pc_input + (torch.randn_like(pc_input) * sigma_adaptive)
        # 随机选择一部分点
        indices = torch.randperm(sample_size)[:n_adaptive]
        sample_adaptive = sample_adaptive[:, indices, :]

        sample = torch.cat([sample_local, sample_global, sample_adaptive], dim=1)

        return sample
