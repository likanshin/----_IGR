import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import torch
import torch.nn.functional as F
from model.network import gradient
from model.sample import Sampler


def adjust_learning_rate(initial_lr, optimizer, iter):
    adjust_lr_every = 400
    lr = initial_lr * ((0.1) ** (iter // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def optimize_latent(points, normals, conf, num_of_iterations, network, lr=1.0e-2, device='cuda', use_amp=False):

    latent_size = conf.get_int('train.latent_size')
    global_sigma = conf.get_float('network.sampler.properties.global_sigma')
    local_sigma = conf.get_float('network.sampler.properties.local_sigma')
    sampler = Sampler.get_sampler(conf.get_string('network.sampler.sampler_type'))(global_sigma, local_sigma)

    latent_lambda = conf.get_float('network.loss.latent_lambda')

    normals_lambda = conf.get_float('network.loss.normals_lambda')

    grad_lambda = conf.get_float('network.loss.lambda')

    div_lambda = conf.get_float('network.loss.div_lambda')

    num_of_points, dim = points.shape

    latent = torch.ones(latent_size).normal_(0, 1 / latent_size).to(device)
    # latent = torch.zeros(latent_size).to(device)

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    # 初始化混合精度训练的 GradScaler
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    for i in range(num_of_iterations):

        sample = sampler.get_points(points.unsqueeze(0)).squeeze()

        latent_all = latent.expand(num_of_points, -1)
        surface_pnts = torch.cat([latent_all, points], dim=1)

        sample_latent_all = latent.expand(sample.shape[0], -1)
        nonsurface_pnts = torch.cat([sample_latent_all, sample], dim=1)

        surface_pnts.requires_grad_()
        nonsurface_pnts.requires_grad_()

        if use_amp:
            surface_pred = network(surface_pnts.float())
            nonsurface_pred = network(nonsurface_pnts.float())

            surface_grad = gradient(surface_pnts, surface_pred)
            nonsurface_grad = gradient(nonsurface_pnts, nonsurface_pred)

            surface_loss = torch.abs(surface_pred).mean()
            grad_norm = nonsurface_grad.norm(2, dim=-1)
            grad_loss = torch.log(1 + ((grad_norm - 1) ** 2)).mean()
            mnfld_grad_norm = surface_grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            grad_dir = surface_grad / mnfld_grad_norm
            normals_loss = (1 - F.cosine_similarity(grad_dir, normals, dim=-1)).mean()
            latent_loss = latent.abs().mean()

            if div_lambda > 0:
                surface_grad_dir = F.normalize(surface_grad, dim=-1)
                nonsurface_grad_dir = F.normalize(nonsurface_grad, dim=-1)
                divergence_loss = (1 - F.cosine_similarity(nonsurface_grad_dir, surface_grad_dir, dim=-1)).mean()
            else:
                divergence_loss = torch.zeros(1, device=points.device)

            loss = surface_loss + latent_lambda * latent_loss + normals_lambda * normals_loss + grad_lambda * grad_loss + div_lambda * divergence_loss
        else:
            surface_pred = network(surface_pnts)
            nonsurface_pred = network(nonsurface_pnts)

            surface_grad = gradient(surface_pnts, surface_pred)
            nonsurface_grad = gradient(nonsurface_pnts, nonsurface_pred)

            surface_loss = torch.abs(surface_pred).mean()
            # DiGS 稳定归一化梯度损失
            grad_norm = nonsurface_grad.norm(2, dim=-1)
            grad_loss = torch.log(1 + ((grad_norm - 1) ** 2)).mean()
            # DiGS 余弦相似度法向量损失
            mnfld_grad_norm = surface_grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            grad_dir = surface_grad / mnfld_grad_norm
            normals_loss = (1 - F.cosine_similarity(grad_dir, normals, dim=-1)).mean()
            latent_loss = latent.abs().mean()

            # 散度正则化代理：表面点与非表面点梯度方向一致性
            if div_lambda > 0:
                surface_grad_dir = F.normalize(surface_grad, dim=-1)
                nonsurface_grad_dir = F.normalize(nonsurface_grad, dim=-1)
                divergence_loss = (1 - F.cosine_similarity(nonsurface_grad_dir, surface_grad_dir, dim=-1)).mean()
            else:
                divergence_loss = torch.zeros(1, device=points.device)

            loss = surface_loss + latent_lambda * latent_loss + normals_lambda * normals_loss + grad_lambda * grad_loss + div_lambda * divergence_loss

        adjust_learning_rate(lr, optimizer, i)

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()

            # 梯度裁剪（需要在 unscaling 之前）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)

            optimizer.step()

        print('latent loss iter {0}:{1}'.format(i, loss.item()))

    return latent.unsqueeze(0)
