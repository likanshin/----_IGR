import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
from time import time
import argparse
import json
import torch
import torch.nn.functional as F
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from utils.plots import plot_surface, plot_cuts
from utils.eval_metrics import compute_all_metrics


class ShapeSpaceRunner:

    def run(self):

        print("running")

        # 初始化混合精度训练的 GradScaler
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP) training")

        with torch.no_grad():
            sample_pnts, _, _ = next(iter(self.train_dataloader))
            coord_min = sample_pnts.min().item()
            coord_max = sample_pnts.max().item()
            print(f"Data coordinate range: [{coord_min:.4f}, {coord_max:.4f}]")
            bbox_min_cfg = self.conf.get_float('train.bbox_min', default=-1.0)
            bbox_max_cfg = self.conf.get_float('train.bbox_max', default=1.0)
            print(f"Configured bbox: [{bbox_min_cfg}, {bbox_max_cfg}]")
            if coord_min < bbox_min_cfg or coord_max > bbox_max_cfg:
                print("WARNING: Data coordinates exceed configured bbox! This will cause hash encoding clamp issues.")

        for epoch in range(self.startepoch, self.nepochs + 1):

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                self.plot_validation_shapes(epoch)

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            if self.grad_warmup_epochs > 0:
                warmup_ratio = 0.1 + 0.9 * min(1.0, epoch / self.grad_warmup_epochs)
                current_grad_lambda = self.grad_lambda * warmup_ratio
            else:
                current_grad_lambda = self.grad_lambda

            if self.div_warmup_epochs > 0:
                div_warmup_ratio = min(1.0, epoch / self.div_warmup_epochs)
                current_div_lambda = self.div_lambda * div_warmup_ratio
            else:
                current_div_lambda = self.div_lambda

            # start epoch
            before_epoch = time()
            for data_index,(mnfld_pnts, normals, indices) in enumerate(self.train_dataloader):

                mnfld_pnts = mnfld_pnts.cuda()

                if self.with_normals:
                    normals = normals.cuda()

                nonmnfld_pnts = self.sampler.get_points(mnfld_pnts)

                mnfld_pnts = self.add_latent(mnfld_pnts, indices)
                nonmnfld_pnts = self.add_latent(nonmnfld_pnts, indices)

                # forward pass

                mnfld_pnts.requires_grad_()
                nonmnfld_pnts.requires_grad_()

                if self.use_amp:
                    mnfld_pred = self.network(mnfld_pnts.float())
                    nonmnfld_pred = self.network(nonmnfld_pnts.float())

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
                    nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

                    mnfld_loss = (mnfld_pred.abs()).mean()

                    grad_norm = nonmnfld_grad.norm(2, dim=-1)
                    grad_loss = torch.log(1 + ((grad_norm - 1) ** 2)).mean()

                    loss = mnfld_loss + current_grad_lambda * grad_loss

                    if self.with_normals:
                        normals = normals.view(-1, 3)
                        mnfld_grad_norm = mnfld_grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        grad_dir = mnfld_grad / mnfld_grad_norm
                        normals_loss = (1 - F.cosine_similarity(grad_dir, normals, dim=-1)).mean()
                        loss = loss + self.normals_lambda * normals_loss
                    else:
                        normals_loss = torch.zeros(1)

                    if current_div_lambda > 0:
                        mnfld_grad_dir = F.normalize(mnfld_grad, dim=-1)
                        n_mnfld = mnfld_grad.shape[0]
                        idx = torch.randperm(nonmnfld_grad.shape[0], device=mnfld_pnts.device)[:n_mnfld]
                        nonmnfld_grad_dir = F.normalize(nonmnfld_grad[idx], dim=-1)
                        divergence_loss = (1 - F.cosine_similarity(nonmnfld_grad_dir, mnfld_grad_dir, dim=-1)).mean()
                        loss = loss + current_div_lambda * divergence_loss
                    else:
                        divergence_loss = torch.zeros(1, device=mnfld_pnts.device)

                    latent_loss = self.latent_size_reg(indices.cuda())

                    loss = loss + self.latent_lambda * latent_loss
                else:
                    mnfld_pred = self.network(mnfld_pnts)
                    nonmnfld_pred = self.network(nonmnfld_pnts)

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
                    nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

                    # manifold loss
                    mnfld_loss = (mnfld_pred.abs()).mean()

                    # DiGS 稳定归一化梯度损失（替代 Eikonal）
                    grad_norm = nonmnfld_grad.norm(2, dim=-1)
                    # 使用 log(1 + x²) 替代 x²/(x+ε)，避免除零
                    grad_loss = torch.log(1 + ((grad_norm - 1) ** 2)).mean()

                    loss = mnfld_loss + current_grad_lambda * grad_loss

                    # normals loss：DiGS 余弦相似度
                    if self.with_normals:
                        normals = normals.view(-1, 3)
                        mnfld_grad_norm = mnfld_grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        grad_dir = mnfld_grad / mnfld_grad_norm
                        normals_loss = (1 - F.cosine_similarity(grad_dir, normals, dim=-1)).mean()
                        loss = loss + self.normals_lambda * normals_loss
                    else:
                        normals_loss = torch.zeros(1)

                    # 散度正则化代理：表面点与非表面点梯度方向一致性
                    if current_div_lambda > 0:
                        mnfld_grad_dir = F.normalize(mnfld_grad, dim=-1)
                        n_mnfld = mnfld_grad.shape[0]
                        idx = torch.randperm(nonmnfld_grad.shape[0], device=mnfld_pnts.device)[:n_mnfld]
                        nonmnfld_grad_dir = F.normalize(nonmnfld_grad[idx], dim=-1)
                        divergence_loss = (1 - F.cosine_similarity(nonmnfld_grad_dir, mnfld_grad_dir, dim=-1)).mean()
                        loss = loss + current_div_lambda * divergence_loss
                    else:
                        divergence_loss = torch.zeros(1, device=mnfld_pnts.device)

                    # latent loss
                    latent_loss = self.latent_size_reg(indices.cuda())

                    loss = loss + self.latent_lambda * latent_loss

                # back propagation

                self.optimizer.zero_grad()

                if self.use_amp:
                    scaler.scale(loss).backward()

                    # 梯度裁剪（需要在 unscaling 之前）
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.lat_vecs, max_norm=self.grad_clip_norm)

                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.lat_vecs, max_norm=self.grad_clip_norm)

                    self.optimizer.step()

                # print status
                if data_index % self.conf.get_int('train.status_frequency') == 0:
                    print('E{} [{}/{}] L:{:.4f} Mnf:{:.4f} Grd:{:.4f}(λ{:.2f}) Lat:{:.4f} Nrm:{:.4f} Div:{:.4f} GN:{:.4f}'.format(
                        epoch, data_index * self.batch_size, len(self.ds),
                        loss.item(), mnfld_loss.item(), grad_loss.item(), current_grad_lambda,
                        latent_loss.item(), normals_loss.item(), divergence_loss.item(),
                        nonmnfld_grad.norm(2, dim=-1).mean().item()))

            after_epoch = time()
            print('epoch time {0}'.format(str(after_epoch-before_epoch)))

    def plot_validation_shapes(self, epoch, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            print('plot validation epoch: ', epoch)

            self.network.eval()
            pnts, normals, idx = next(iter(self.eval_dataloader))
            pnts = utils.to_cuda(pnts)

            pnts = self.add_latent(pnts, idx)
            latent = self.lat_vecs[idx[0]]

            shapename = str.join('_', self.ds.get_info(idx))

            pred_mesh = plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         latent=latent,
                         path=self.plots_dir,
                         epoch=epoch,
                         shapename=shapename,
                         **self.conf.get_config('plot'))

            if with_cuts:
                plot_cuts(points=pnts,
                          decoder=self.network,
                          latent=latent,
                          path=self.plots_dir,
                          epoch=epoch,
                          near_zero=False)

            # 计算评估指标
            if self.compute_eval_metrics and pred_mesh is not None:
                try:
                    pred_mesh_path = os.path.join(self.plots_dir, 'igr_{0}_{1}.ply'.format(epoch, shapename))
                    gt_mesh_path = self.ds.get_gt_mesh_path(idx[0])
                    
                    if gt_mesh_path is None or not os.path.exists(pred_mesh_path):
                        print('Skipping evaluation: gt_mesh or pred_mesh not found')
                    else:
                        metrics = compute_all_metrics(
                            pred_mesh_path, 
                            gt_mesh_path, 
                            num_samples=self.eval_num_samples,
                            f_threshold=self.eval_f_threshold
                        )
                        
                        print('Evaluation Metrics (Epoch {0}):'.format(epoch))
                        print('  Chamfer Distance: {0:.6f}'.format(metrics['chamfer_distance']))
                        print('  F-Score: {0:.6f}'.format(metrics['f_score']))
                        print('  Normal Consistency: {0:.6f}'.format(metrics['normal_consistency']))
                        
                        metrics_path = os.path.join(self.plots_dir, 'eval_metrics.json')
                        if os.path.exists(metrics_path):
                            with open(metrics_path, 'r') as f:
                                all_metrics = json.load(f)
                        else:
                            all_metrics = []
                        
                        metrics['epoch'] = epoch
                        metrics['shapename'] = shapename
                        all_metrics.append(metrics)
                        
                        with open(metrics_path, 'w') as f:
                            json.dump(all_metrics, f, indent=2)
                        
                except Exception as e:
                    print('Failed to compute evaluation metrics: {0}'.format(str(e)))

    def __init__(self,**kwargs):

        # config setting

        self.home_dir = os.path.abspath(os.pardir)

        if type(kwargs['conf']) == str:
            self.conf_filename = './shapespace/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings

        self.GPU_INDEX = kwargs['gpu_index']

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        # settings for loading an existing experiment

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = self.timestamp
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.cur_exp_dir))

        self.plots_dir = os.path.join(self.expdir, self.cur_exp_dir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.expdir, self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.expdir, self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.latent_codes_subdir = "LatentCodes"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path,self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.latent_codes_subdir))

        self.nepochs = kwargs['nepochs']

        self.batch_size = kwargs['batch_size']

        if self.num_of_gpus > 0:
            self.batch_size *= self.num_of_gpus

        self.parallel = self.num_of_gpus > 1

        self.global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        self.local_sigma = self.conf.get_float('network.sampler.properties.local_sigma')
        self.sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(self.global_sigma, self.local_sigma)

        train_split_file = './splits/{0}'.format(kwargs['split_file'])

        with open(train_split_file, "r") as f:
            train_split = json.load(f)

        self.d_in = self.conf.get_int('train.d_in')

        # latent preprocessing

        self.latent_size = self.conf.get_int('train.latent_size')

        self.latent_lambda = self.conf.get_float('network.loss.latent_lambda')
        self.grad_lambda = self.conf.get_float('network.loss.lambda')
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')
        self.div_lambda = self.conf.get_float('network.loss.div_lambda')
        self.grad_clip_norm = self.conf.get_float('network.loss.grad_clip_norm', default=5.0)
        self.grad_warmup_epochs = self.conf.get_int('network.loss.grad_warmup_epochs', default=0)
        self.div_warmup_epochs = self.conf.get_int('network.loss.div_warmup_epochs', default=50)
        self.use_amp = self.conf.get_bool('train.use_amp', default=False)
        self.compute_eval_metrics = self.conf.get_bool('train.compute_eval_metrics', default=False)
        self.eval_num_samples = self.conf.get_int('train.eval_num_samples', default=100000)
        self.eval_f_threshold = self.conf.get_float('train.eval_f_threshold', default=0.01)

        self.with_normals = self.normals_lambda > 0

        self.ds = utils.get_class(self.conf.get_string('train.dataset'))(split=train_split,
                                                                         with_normals=self.with_normals,
                                                                         with_gt=self.compute_eval_metrics,
                                                                         dataset_path=self.conf.get_string(
                                                                             'train.dataset_path'),
                                                                         points_batch=kwargs['points_batch'],
                                                                         )

        self.num_scenes = len(self.ds)

        self.train_dataloader = torch.utils.data.DataLoader(self.ds,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=kwargs['threads'], drop_last=True, pin_memory=True)
        self.eval_dataloader = torch.utils.data.DataLoader(self.ds,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=0, drop_last=True)

        bbox_min = self.conf.get_float('train.bbox_min', default=-1.0)
        bbox_max = self.conf.get_float('train.bbox_max', default=1.0)
        self.network = utils.get_class(self.conf.get_string('train.network_class'))(
            d_in=(self.d_in+self.latent_size),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            **self.conf.get_config('network.inputs'))

        if self.parallel:
            self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        # optimizer and latent settings

        self.startepoch = 0

        self.lat_vecs = torch.zeros(self.num_scenes, self.latent_size).cuda()
        self.lat_vecs.requires_grad_()

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
                {
                    "params": self.lat_vecs,
                    "lr": self.lr_schedules[1].get_learning_rate(0)
                },
            ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            data = torch.load(os.path.join(old_checkpnts_dir, self.latent_codes_subdir, str(kwargs['checkpoint']) + '.pth'))
            self.lat_vecs = data["latent_codes"].cuda()

            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"], strict=False)

            data = torch.load(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def latent_size_reg(self, indices):
        latents = torch.index_select(self.lat_vecs, 0, indices)
        latent_loss = latents.norm(dim=1).mean()
        return latent_loss

    def get_learning_rate_schedules(self,schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def add_latent(self, points, indices):
        batch_size, num_of_points, dim = points.shape
        points = points.reshape(batch_size * num_of_points, dim)
        latent_inputs = torch.zeros(0).cuda()

        for ind in indices.numpy():
            latent_ind = self.lat_vecs[ind]
            latent_repeat = latent_ind.expand(num_of_points, -1)
            latent_inputs = torch.cat([latent_inputs, latent_repeat], 0)
        points = torch.cat([latent_inputs, points], 1)
        return points

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self,epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "latent_codes": self.lat_vecs},
            os.path.join(self.checkpoints_path, self.latent_codes_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "latent_codes": self.lat_vecs},
            os.path.join(self.checkpoints_path, self.latent_codes_subdir, "latest.pth"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--points_batch', type=int, default=8000, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='dfaust_setup.conf')
    parser.add_argument('--expname', type=str, default='dfuast_shapespace')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU ignore]')
    parser.add_argument('--threads', type=int, default=32, help='num of threads for data loader')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)
    parser.add_argument('--split', default='dfaust/train_all.json', type=str)

    args = parser.parse_args()

    trainrunner = ShapeSpaceRunner(
            conf=args.conf,
            batch_size=args.batch_size,
            points_batch=args.points_batch,
            nepochs=args.nepoch,
            expname=args.expname,
            gpu_index=args.gpu,
            threads=args.threads,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint,
            split_file=args.split
    )

    trainrunner.run()
