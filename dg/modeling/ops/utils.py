
import torch

























def create_onehot(label, num_classes):
    """Create one-hot tensor.

    We suggest using nn.functional.one_hot.

    Args:
        label (torch.Tensor): 1-D tensor.
        num_classes (int): number of classes.
    """
    onehot = torch.zeros(label.shape[0], num_classes)
    return onehot.scatter(1, label.unsqueeze(1).data.cpu(), 1)
