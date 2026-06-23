import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import argparse
import json
import utils.general as utils
import torch
import numpy as np
import utils.plots as plt
from pyhocon import ConfigFactory
from shapespace.latent_optimizer import optimize_latent


def interpolate(network, interval, experiment_directory, checkpoint, split_file, epoch, resolution, uniform_grid, device):

    with open(split_file, "r") as f:
        split = json.load(f)

    ds = utils.get_class(conf.get_string('train.dataset'))(split=split, dataset_path=conf.get_string('train.dataset_path'), with_normals=True)

    points_1, normals_1, index_1 = ds[0]
    points_2, normals_2, index_2 = ds[1]

    pnts = torch.cat([points_1, points_2], dim=0).to(device)

    name_1 = str.join('_', ds.get_info(0))
    name_2 = str.join('_', ds.get_info(1))

    name = name_1 + '_and_' + name_2

    utils.mkdir_ifnotexists(os.path.join(experiment_directory, 'interpolate'))
    utils.mkdir_ifnotexists(os.path.join(experiment_directory, 'interpolate', str(checkpoint)))
    utils.mkdir_ifnotexists(os.path.join(experiment_directory, 'interpolate', str(checkpoint), name))

    my_path = os.path.join(experiment_directory, 'interpolate', str(checkpoint), name)

    latent_1 = optimize_latent(points_1.to(device), normals_1.to(device), conf, 1200, network, 5e-3, device)
    latent_2 = optimize_latent(points_2.to(device), normals_2.to(device), conf, 1200, network, 5e-3, device)

    pnts = torch.cat([latent_1.repeat(pnts.shape[0], 1), pnts], dim=-1)

    with torch.no_grad():
        network.eval()

        for alpha in np.linspace(0,1, interval):

            latent = (latent_1 * (1-alpha)) + (latent_2 * alpha)

            plt.plot_surface(with_points=False,
                             points=pnts,
                             decoder=network,
                             latent=latent,
                             path=my_path,
                             epoch=epoch,
                             shapename=str(alpha),
                             resolution=resolution,
                             mc_value=0,
                             is_uniform_grid=uniform_grid,
                             verbose=True,
                             save_html=False,
                             save_ply=True,
                             overwrite=True,
                             connected=True)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--interval",
        "-i",
        dest="interval",
        default=3,
        type=int
    )

    arg_parser.add_argument(
        "--gpu",
        "-g",
        dest="gpu_num",
        required=False,
        default='0'
    )

    arg_parser.add_argument(
        "--timestamp",
        "-t",
        dest="timestamp",
        default='latest',
        required=False,
    )

    arg_parser.add_argument(
        "--conf",
        "-f",
        dest="conf",
        default='dfaust_setup.conf',
        required=False,
    )

    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split",
        default='dfaust/interpolate.json',
        required=False,
    )

    arg_parser.add_argument(
        "--exp-name",
        "-e",
        dest="exp_name",
        required=True,
        help="experiment name",
    )

    arg_parser.add_argument(
        "--exps-dir",
        dest="exps_dir",
        required=False,
        default='exps'
    )

    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="epoch",
        default='latest',
        help="The checkpoint to test.",
    )

    arg_parser.add_argument(
        "--resolution",
        "-r",
        dest="resolution",
        help='resolution of marching cube grid',
        default=256,
        #default=128,
        type=int
    )

    arg_parser.add_argument(
        "--uniform-grid",
        "-u",
        dest="uniform_grid",
        help='use uniform grid in marching cube or non uniform',
        default=False
    )

    cur_dir = os.path.abspath('dfaust')

    args = arg_parser.parse_args()

    code_path = os.path.abspath(os.path.curdir)
    exps_path = os.path.join(os.path.abspath(os.path.pardir), args.exps_dir)

    if args.gpu_num != 'ignore':
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(args.gpu_num)

    conf = ConfigFactory.parse_file(os.path.join(code_path, 'shapespace', args.conf))

    experiment_directory = os.path.join(exps_path, args.exp_name)

    if args.timestamp == 'latest':
        timestamps = os.listdir(experiment_directory)
        timestamp = sorted(timestamps)[-1]
    else:
        timestamp = args.timestamp

    experiment_directory = os.path.join(experiment_directory, timestamp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_model_state = torch.load(os.path.join(experiment_directory, 'checkpoints', 'ModelParameters', args.epoch + ".pth"), map_location=device)
    saved_model_epoch = saved_model_state["epoch"]
    with_normals = conf.get_float('network.loss.normals_lambda') > 0

    # 自动检测 checkpoint 类型（兼容 ImplicitNet 和 HashGridImplicitNet）
    state_dict_keys = [k.replace('module.', '') for k in saved_model_state["model_state_dict"].keys()]
    if any(k.startswith('lin') for k in state_dict_keys):
        network = utils.get_class('model.network.ImplicitNet')(
            d_in=conf.get_int('train.latent_size') + conf.get_int('train.d_in'),
            dims=[512, 512, 512, 512, 512, 512, 512, 512],
            skip_in=[4], geometric_init=True, radius_init=1, beta=100)
        print('Detected ImplicitNet checkpoint')
    else:
        network = utils.get_class(conf.get_string('train.network_class'))(
            d_in=conf.get_int('train.latent_size') + conf.get_int('train.d_in'),
            **conf.get_config('network.inputs'))
        print('Detected HashGridImplicitNet checkpoint')

    network.load_state_dict({k.replace('module.', ''): v for k, v in saved_model_state["model_state_dict"].items()})
    split_file = os.path.join(code_path, 'splits', args.split)

    interpolate(
        network=network.to(device),
        interval=args.interval,
        experiment_directory=experiment_directory,
        checkpoint=saved_model_epoch,
        split_file=split_file,
        epoch=saved_model_epoch,
        resolution=args.resolution,
        uniform_grid=args.uniform_grid,
        device=device
    )
