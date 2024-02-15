import os
import sys

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from utils.util import instantiate_from_config, rank_zero_print
from utils.config import get_parser, parse_configs


if __name__ == "__main__":
    
    sys.path.append(os.getcwd())
    # load parser
    parser = get_parser()
    # parsing configs
    config, opt, trainer_opt, trainer_kwargs, trainer_config = parse_configs(parser)
    seed_everything(opt.seed)
    # data
    data = instantiate_from_config(config.data)
    # model
    model = instantiate_from_config(config.model)
    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = 1 if not "gpus" in trainer_config else len(trainer_config.gpus.strip(",").split(','))
    if opt.scale_lr:
        model.learning_rate = ngpu * bs * base_lr 
        rank_zero_print("Setting learning rate to {:.2e} =  {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(model.learning_rate, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")
    # trainer
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    if opt.train:
        trainer.fit(model, data)
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)