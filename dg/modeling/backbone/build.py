from dg.utils import Registry, check_availability

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(name, verbose=True, **kwargs):
    avai_backbones = BACKBONE_REGISTRY.registered_names()
    # print("Available Backbones: ", avai_backbones)
    check_availability(name, avai_backbones)
    # if verbose:
    #     print("Backbone: {}".format(name))

    return BACKBONE_REGISTRY.get(name)(**kwargs)
