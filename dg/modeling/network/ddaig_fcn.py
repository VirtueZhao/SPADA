"""
Credit to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
import functools
import torch
import torch.nn as nn
from torch.nn import functional as F

from .build import NETWORK_REGISTRY


def init_network_weights(model, init_type="normal", gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method {} is not implemented".format(init_type)
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("InstanceNorm2d") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(
            "normalization layer [%s] is not found" % norm_type
        )
    return norm_layer


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().ResnetBlock.__init__()")
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().ResnetBlock.__init__()")

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().ResnetBlock.__init__().build_conv_block()")
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                "padding [%s] is not implemented" % padding_type
            )

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                "padding [%s] is not implemented" % padding_type
            )
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().ResnetBlock.__init__().build_conv_block()")
        return nn.Sequential(*conv_block)

    def forward(self, x):
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G.ddaig_fcn.forward().ResnetBlock.forward()")
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G.ddaig_fcn.forward().ResnetBlock.forward()")
        return x + self.conv_block(x)


















































class FCN(nn.Module):
    """Fully convolutional network."""

    def __init__(
        self,
        input_nc,
        output_nc,
        nc=32,
        n_blocks=3,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        padding_type="reflect",
        gctx=True,
        stn=False,
        image_size=32
    ):
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__()")
        super().__init__()
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().build_backbone")
        backbone = []

        p = 0
        if padding_type == "reflect":
            backbone += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            backbone += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError
        backbone += [nn.Conv2d(input_nc, nc, kernel_size=3, stride=1, padding=p, bias=False)]
        backbone += [norm_layer(nc)]
        backbone += [nn.ReLU(True)]

        for _ in range(n_blocks):
            backbone += [
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=False
                )
            ]
        self.backbone = nn.Sequential(*backbone)
        # print(self.backbone)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().build_backbone")

        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().build_gctx_fusion")
        self.gctx_fusion = None
        if gctx:
            self.gctx_fusion = nn.Sequential(
                nn.Conv2d(2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(nc),
                nn.ReLU(True)
            )
        # print(self.gctx_fusion)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().build_gctx_fusion")

        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().build_regress")
        self.regress = nn.Sequential(
            nn.Conv2d(nc, output_nc, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
        )
        # print(self.regress)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__().build_regress")

        self.locnet = None
        if stn:
            self.locnet = LocNet(input_nc, nc=nc, n_blocks=n_blocks, image_size=image_size)

        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.FCN.__init__()")















    def forward(self, x, lmda=1.0, return_p=False, return_stn_output=False):
        """
        Args:
            x (torch.Tensor): input mini-batch.
            lmda (float): multiplier for perturbation.
            return_p (bool): return perturbation.
            return_stn_output (bool): return the output of stn.
        """
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G.ddaig_fcn.forward()")
        theta = None
        if self.locnet is not None:
            x, theta = self.stn(x)

        input = x
        x = self.backbone(x)
        # print(self.gctx_fusion)
        if self.gctx_fusion is not None:
            c = F.adaptive_avg_pool2d(x, (1, 1))
            # print("c.shape: {}".format(c.shape))
            c = c.expand_as(x)
            # print("x.shape: {}".format(x.shape))
            # print("c.shape: {}".format(c.shape))
            x = torch.cat([x, c], 1)
            # print("x.shape: {}".format(x.shape))
            x = self.gctx_fusion(x)
            # print("x.shape: {}".format(x.shape))
            # print(x)

        p = self.regress(x)
        # print(p)
        x_p = input + lmda * p
        # print(x_p)

        if return_stn_output:
            return x_p, p, input

        if return_p:
            return x_p, p

        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G.ddaig_fcn.forward()")
        return x_p


@NETWORK_REGISTRY.register()
def fcn_3x32_gctx(**kwargs):
    norm_layer = get_norm_layer(norm_type="instance")
    net = FCN(3, 3, nc=32, n_blocks=3, norm_layer=norm_layer)
    init_network_weights(net, init_type="normal", gain=0.02)
    return net


@NETWORK_REGISTRY.register()
def fcn_3x64_gctx(**kwargs):
    print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx")
    print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.get_norm_layer()")
    norm_layer = get_norm_layer(norm_type="instance")
    # print(norm_layer)
    print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.get_norm_layer()")
    net = FCN(3, 3, nc=64, n_blocks=3, norm_layer=norm_layer)
    # print(net)
    print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.init_network_weights()")
    init_network_weights(net, init_type="normal", gain=0.02)
    print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx.init_network_weights()")
    print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network().fcn_3x64_gctx")
    return net
