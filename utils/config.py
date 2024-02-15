import os
import glob
import datetime
import importlib
import argparse
import pytorch_lightning as pl

from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


from utils.logger import SetupCallback, ImageLogger
from utils.util import rank_zero_print, instantiate_from_config

def get_parser(**parser_kwargs):
    # arguments
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('--finetune_from', type=str, nargs='?', default='', help='path to checkpoint to load model state from')
    parser.add_argument('-n', '--name', type=str, const=True, nargs='?', default='', help='postfix for logdir')
    parser.add_argument('-r', '--resume', type=str, const=True, nargs='?', default='', help='resume from logdir or checkpoint in logdir')
    parser.add_argument('-b', '--base', nargs='*', default=list(), metavar='base_config.yaml', help='paths to base configs')
    parser.add_argument('-t', '--train', type=str2bool, const=True, nargs='?', default=False, help='train')
    parser.add_argument('--no_test', type=str2bool, const=True, nargs='?', default=False, help='disable test')
    parser.add_argument('-d', '--debug', type=str2bool, const=True, default=False, nargs="?", help="enable post-mortem debugging")
    parser.add_argument('-s', '--seed', type=int, default=23, help='seed for seed everything')
    parser.add_argument('-l', '--logdir', type=str, default='logs', help='directory for logging dat shit')
    parser.add_argument('--scale_lr', type=str2bool, const=True, nargs='?', default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument('--resolution', type=int, default=512, help="resolution of image")
    parser.add_argument('--results', type=str, default=None, nargs='?', help='paths to save the results')

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# set checkpoint and training dir
def set_check_dir(opt, now=None):
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    return logdir, ckptdir, cfgdir

# parsing configs from yaml
def parse_configs(parser):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # set training / pre-trained folder
    logdir, ckptdir, cfgdir = set_check_dir(opt, now)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # load the finetune or resume file
    if opt.finetune_from !='' or opt.resume_from_checkpoint is not None:
        config['model']['params']['ckpt_path'] = opt.finetune_from or opt.resume_from_checkpoint
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "gpu"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        if version.parse(pl.__version__) >= version.parse('1.7.0'):
            trainer_config["devices"] = trainer_config["gpus"]
            del trainer_config["gpus"]
        rank_zero_print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # configure the trainer and callbacks
    trainer_kwargs = dict()
    
    # log config
    logger_cfg = lightning_config.get("logger", OmegaConf.create())
    logger_cfg['params']['name'] = logdir.split('/')[-1]
    logger_cfg['params']['save_dir'] = logdir
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model checkpoint for config
    modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
    modelckpt_cfg['params']['dirpath'] = ckptdir
    modelckpt_cfg['params']['filename'] = '{epoch:06}'
    
    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "utils.logger.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
                "debug": opt.debug,
            }
        },
        "learning_rate_logger": {
            "target": "utils.config.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if lightning_config.get("metrics_over_trainsteps_checkpoint", OmegaConf.create()):
        rank_zero_print('Saving checkpoints every n train steps without deleting.')
        default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": ckptdir,
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         "save_top_k": 3,
                         "every_n_train_steps": 10000,
                         "save_weights_only": True,
                         "monitor": "train/loss"
                     }
                }
            }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]

    lightning_config.trainer = trainer_config

    return config, opt, trainer_opt, trainer_kwargs, trainer_config
