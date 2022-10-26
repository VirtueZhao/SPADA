import argparse
import torch

from dg.utils import setup_logger, set_random_seed, collect_env_info
from dg.config import get_cfg_default
from dg.engine import build_trainer


def print_args(args, cfg):
    # print("***************")
    # print("** Arguments **")
    # print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    # print("************")
    print("** Config **")
    # print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.dataset_path:
        cfg.DATASET.PATH = args.dataset_path

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    # if args.resume:
    #     cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domain:
        cfg.DATASET.TARGET_DOMAIN = args.target_domain

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    pass


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.config_path_dataset:
        cfg.merge_from_file(args.config_path_dataset)

    # 2. From the method config file
    if args.config_path_trainer:
        cfg.merge_from_file(args.config_path_trainer)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    # cfg.merge_from_list(args.opts)

    # 5. Specify GPU
    cfg.GPU = args.GPU
    cfg.OPTIM.MAX_EPOCH = args.max_epoch
    cfg.cl_loss = args.cl_loss
    cfg.OPTIM.LR = args.lr
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))
    print("+Calling train.build_trainer()")
    trainer = build_trainer(cfg)
    print("-Closing: train.build_trainer()")
    print()

    # if args.eval_only:
    #     trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #     trainer.test()
    #     return

    if not args.no_train:
        print("+Calling: train.trainer.train()")
        trainer.train()
        print("-Closing: train.trainer.train()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU",
        type=str,
        default="0",
        help="specify GPU"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="",
        help="path to dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="",
        help="output directory"
    )
    # parser.add_argument(
    #     "--resume",
    #     type=str,
    #     default="",
    #     help="checkpoint directory (from which the training resumes)"
    # )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--cl_loss",
        type=float
    )
    parser.add_argument(
        "--lr",
        type=float
    )
    parser.add_argument(
        "--batch_size",
        type=int
    )
    parser.add_argument(
        "--source_domains",
        type=str,
        nargs="+",
        help="source domain for domain generalization"
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        nargs="+",
        help="target domain for domain generalization"
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        help="data augmentation methods"
    )
    parser.add_argument(
        "--config_path_trainer",
        type=str,
        # default="",
        help="trainer config file path"
    )
    parser.add_argument(
        "--config_path_dataset",
        type=str,
        # default="",
        help="dataset config file path"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        # default="",
        help="name of trainers"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        # default="",
        help="name of CNN backbone"
    )
    parser.add_argument(
        "--head",
        type=str,
        # default="",
        help="name of head"
    )
    # parser.add_argument(
    #     "--eval-only",
    #     action="store_true",
    #     help="evaluation only"
    # )
    # parser.add_argument(
    #     "--model-dir",
    #     type=str,
    #     default="",
    #     help="load model from this directory for eval-only mode"
    # )
    # parser.add_argument(
    #     "--load-epoch",
    #     type=int,
    #     help="load model weights at this epoch for evaluation"
    # )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="do not call trainer.train()"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=25,
        help="only positive value enables a fixed seed"
    )
    # parser.add_argument(
    #     "opts",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    #     help="modify config options using the command line"
    # )

    args = parser.parse_args()
    main(args)
