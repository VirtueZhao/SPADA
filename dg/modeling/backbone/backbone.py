import torch.nn as nn


class Backbone(nn.Module):

    def __init__(self):
        print("+Calling: ddaig.__init__().SimpleTrainer.__init__().DDAIG.build_model().SimpleNet.__init__().build_backbone().resnet18().ResNet.__init__().Backbone.__init__()")
        super().__init__()
        print("-Closing: ddaig.__init__().SimpleTrainer.__init__().DDAIG.build_model().SimpleNet.__init__().build_backbone().resnet18().ResNet.__init__().Backbone.__init__()")

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features
