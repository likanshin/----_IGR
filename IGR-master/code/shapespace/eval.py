import argparse
import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import json
import utils.general as utils
import torch
from pyhocon import ConfigFactory
import utils.plots as plt
from shapespace.latent_optimizer import optimize_latent


def evaluate(network, experiment_directory, conf, checkpoint, split_file, epoch, resolution, uniform_grid):

    my_path = os.path.join(experiment_directory, 'evaluation', str(checkpoint))

    utils.mkdir_ifnotexists(os.path.join(experiment_directory, 'evaluation'))
    utils.mkdir_ifnotexists(my_path)

    with open(split_file, "r") as f:
        split = json.load(f)

    ds = utils.get_class(conf.get_string('train.dataset'))(split=split, dataset_path=conf.get_string('train.dataset_path'), with_normals=True)

    total_files = len(ds)
    print("total files : {0}".format(total_files))
    counter = 0
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=1, drop_last=False, pin_memory=True)

    for (input_pc, normals, index) in dataloader:

        input_pc = input_pc.cuda().squeeze()
        normals = normals.cuda().squeeze()

        print(counter)
        counter = counter + 1

        network.train()

        latent = optimize_latent(input_pc, normals, conf, 800, network, lr=5e-3)

        all_latent = latent.repeat(input_pc.shape[0], 1)

        points = torch.cat([all_latent,input_pc], dim=-1)

        shapename = str.join('_', ds.get_info(index))

        with torch.no_grad():

            network.eval()

            plt.plot_surface(with_points=True,
                             points=points,
                             decoder=network,
                             latent=latent,
                             path=my_path,
                             epoch=epoch,
                             shapename=shapename,
                             resolution=resolution,
                             mc_value=0,
                             is_uniform_grid=uniform_grid,
                             verbose=True,
                             save_html=True,
                             save_ply=True,
                             overwrite=True,
                             connected=True)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--gpu",
        "-g",
        dest="gpu_num",
        required=False,
        default='ignore'
    )

    arg_parser.add_argument(
        "--exps-dir",
        dest="exps_dir",
        required=False,
        default='exps'
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
        default='dfaust/test_models.json',
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
        "--checkpoint",
        "-c",
        dest="epoch",
        default='latest',
        help="checkpoint to test.",
    )

    arg_parser.add_argument(
        "--resolution",
        "-r",
        dest="resolution",
        help='resolution of marching cube grid',
        #default=256,
        default=128,
        type=int
    )

    arg_parser.add_argument(
        "--uniform-grid",
        "-u",
        dest="uniform_grid",
        help='use uniform grid in marching cube or non uniform',
        default=False
    )

    print('evaluating')

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
    saved_model_state = torch.load(
        os.path.join(experiment_directory, 'checkpoints', 'ModelParameters', args.epoch + ".pth"),
        map_location='cuda' if torch.cuda.is_available() else 'cpu')
    saved_model_epoch = saved_model_state["epoch"]

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

    network.load_state_dict({k.replace('module.', ''): v for k, v in
                             saved_model_state["model_state_dict"].items()})

    split_file = os.path.join(code_path, 'splits', args.split)

    evaluate(
        network=network.cuda(),
        experiment_directory=experiment_directory,
        conf=conf,
        checkpoint=saved_model_epoch,
        split_file=split_file,
        epoch=saved_model_epoch,
        resolution=args.resolution,
        uniform_grid=args.uniform_grid
    )


