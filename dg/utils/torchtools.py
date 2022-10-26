"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""


import os.path as osp
import shutil

from collections import OrderedDict
import torch
import torch.nn as nn

from .tools import mkdir_if_missing













def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name=""
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))

    # save current model name
    checkpoint_file = osp.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(osp.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))











































































































































































def count_num_param(model):
    r"""Count number of parameters in a model.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> model_size = count_num_param(model)
    """
    return sum(p.numel() for p in model.parameters())





















































def init_network_weights(model, init_type="normal", gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)
