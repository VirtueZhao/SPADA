from dg.utils import Registry, check_availability

NETWORK_REGISTRY = Registry("NETWORK")


def build_network(name, verbose=True, **kwargs):
    avai_models = NETWORK_REGISTRY.registered_names()
    # print("Available Network:", avai_models)
    check_availability(name, avai_models)
    # if verbose:
    #     print("Network: {}".format(name))
    return NETWORK_REGISTRY.get(name)(**kwargs)
