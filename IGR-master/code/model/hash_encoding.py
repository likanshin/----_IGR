import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HashGridEncoding(nn.Module):
    """
    纯 PyTorch 实现的多分辨率哈希编码
    参考 Instant-NGP 的 HashGrid 编码思想
    """
    def __init__(self, 
                 n_input_dims=3,
                 n_levels=16, 
                 n_features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 per_level_scale=2.0,
                 bbox_min=-1.0,
                 bbox_max=1.0):
        super().__init__()
        
        self.n_input_dims = n_input_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.bbox_min_val = bbox_min
        self.bbox_max_val = bbox_max
        
        self.level_resolutions = []
        for i in range(n_levels):
            resolution = int(np.floor(base_resolution * (per_level_scale ** i)))
            self.level_resolutions.append(resolution)
        
        hashmap_size = 2 ** log2_hashmap_size
        
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(hashmap_size, n_features_per_level) * 0.1)
            for _ in range(n_levels)
        ])
        
        self.n_output_dims = n_levels * n_features_per_level
    
    def hash_function(self, coords, hashmap_size):
        primes = [1, 2654435761, 805459861]
        hashed = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        for i in range(coords.shape[-1]):
            x_i = coords[..., i].long()
            hashed = hashed ^ ((x_i * primes[i]) & 0x7FFFFFFF)
        return (hashed % hashmap_size).long()
    
    def forward(self, x):
        x = x.float()
        
        bbox_range = self.bbox_max_val - self.bbox_min_val
        x_normalized = (x - self.bbox_min_val) / (bbox_range + 1e-8)
        x_normalized = x_normalized.clamp(0.0, 1.0)
        
        encoded_levels = []
        
        for level in range(self.n_levels):
            resolution = self.level_resolutions[level]
            hash_table = self.hash_tables[level]
            
            scaled_coords = x_normalized * (resolution - 1)
            
            with torch.no_grad():
                floor_coords = torch.floor(scaled_coords).long()
                ceil_coords = torch.min(floor_coords + 1, torch.tensor(resolution - 1, device=x.device))
            
            alpha = (scaled_coords - floor_coords.float()).clamp(0.0, 1.0)
            
            corners = []
            weights = []
            
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        with torch.no_grad():
                            cx = floor_coords[:, 0] if dx == 0 else ceil_coords[:, 0]
                            cy = floor_coords[:, 1] if dy == 0 else ceil_coords[:, 1]
                            cz = floor_coords[:, 2] if dz == 0 else ceil_coords[:, 2]
                            corner = torch.stack([cx, cy, cz], dim=-1).clamp(0, resolution - 1)
                        corners.append(corner)
                        
                        wx = (1 - alpha[:, 0]) if dx == 0 else alpha[:, 0]
                        wy = (1 - alpha[:, 1]) if dy == 0 else alpha[:, 1]
                        wz = (1 - alpha[:, 2]) if dz == 0 else alpha[:, 2]
                        weight = wx * wy * wz
                        weights.append(weight)
            
            features = []
            for corner in corners:
                with torch.no_grad():
                    hash_indices = self.hash_function(corner, hash_table.shape[0])
                feature = hash_table[hash_indices]
                features.append(feature)
            
            level_feature = torch.zeros_like(features[0])
            for feature, weight in zip(features, weights):
                level_feature = level_feature + feature * weight.unsqueeze(-1)
            
            encoded_levels.append(level_feature)
        
        return torch.cat(encoded_levels, dim=-1)


class HashGridImplicitNet(nn.Module):
    """
    双路径哈希编码隐式网络 (Instant-NGP + DiGS)
    空间路径和 latent 路径分别处理，确保空间坐标有足够的梯度
    """
    def __init__(self,
                 d_in,
                 n_levels=16,
                 n_features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 per_level_scale=2.0,
                 hidden_dim=64,
                 num_layers=2,
                 geometric_init=True,
                 radius_init=1,
                 num_output_layers=2,
                 bbox_min=-1.0,
                 bbox_max=1.0):
        super().__init__()
        
        self.d_in = d_in
        self.latent_size = max(0, d_in - 3)
        
        self.encoding = HashGridEncoding(
            n_input_dims=3,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
            bbox_min=bbox_min,
            bbox_max=bbox_max
        )
        
        spatial_layers = []
        spatial_layers.append(nn.Linear(self.encoding.n_output_dims, hidden_dim))
        spatial_layers.append(nn.Softplus(beta=100))
        for _ in range(num_layers - 1):
            spatial_layers.append(nn.Linear(hidden_dim, hidden_dim))
            spatial_layers.append(nn.Softplus(beta=100))
        self.spatial_net = nn.Sequential(*spatial_layers)

        if self.latent_size > 0:
            latent_layers = []
            latent_layers.append(nn.Linear(self.latent_size, hidden_dim))
            latent_layers.append(nn.Softplus(beta=100))
            for _ in range(num_layers - 1):
                latent_layers.append(nn.Linear(hidden_dim, hidden_dim))
                latent_layers.append(nn.Softplus(beta=100))
            self.latent_net = nn.Sequential(*latent_layers)

        fusion_dim = hidden_dim * 2 if self.latent_size > 0 else hidden_dim
        output_layers = []
        output_layers.append(nn.Linear(fusion_dim, hidden_dim))
        output_layers.append(nn.Softplus(beta=100))
        for _ in range(num_output_layers - 1):
            output_layers.append(nn.Linear(hidden_dim, hidden_dim))
            output_layers.append(nn.Softplus(beta=100))
        output_layers.append(nn.Linear(hidden_dim, 1))
        self.output_net = nn.Sequential(*output_layers)

        if geometric_init:
            last_layer = self.output_net[-1]
            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=1e-4)
            torch.nn.init.zeros_(last_layer.bias)
    
    def forward(self, input):
        """
        input: [batch_size, d_in] 
               前 latent_size 维是 latent code，后 3 维是空间坐标
        """
        if self.latent_size > 0:
            latent = input[:, :self.latent_size]
            coords = input[:, self.latent_size:]
        else:
            latent = None
            coords = input
        
        # 空间路径
        encoded = self.encoding(coords)
        spatial_feat = self.spatial_net(encoded)
        
        # latent 路径
        if latent is not None:
            latent_feat = self.latent_net(latent)
            combined = torch.cat([spatial_feat, latent_feat], dim=-1)
        else:
            combined = spatial_feat
        
        return self.output_net(combined)
