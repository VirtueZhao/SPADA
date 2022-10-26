import torch
from torch.nn import functional as F

from dg.engine import TRAINER_REGISTRY, TrainerX
from dg.metrics import compute_accuracy
from dg.utils.tools import gradient_magnitude
from dg.utils.curriculum import curriculum_example
__all__ = ["DomainMix"]


@TRAINER_REGISTRY.register()
class DomainMix(TrainerX):
    """DomainMix.

    Dynamic Domain Generalization.

    https://github.com/MetaVisionLab/DDG
    """

    def __init__(self, cfg):
        super(DomainMix, self).__init__(cfg)
        self.mix_type = cfg.TRAINER.DOMAINMIX.TYPE
        self.alpha = cfg.TRAINER.DOMAINMIX.ALPHA
        self.beta = cfg.TRAINER.DOMAINMIX.BETA
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def forward_backward(self, batch):
        input, label_a, label_b, lam, img_id = self.parse_batch_train(batch)
        input.requires_grad = True

        output = self.model(input)

        loss = lam * F.cross_entropy(
            output, label_a
        ) + (1-lam) * F.cross_entropy(output, label_b)

        # print("Current Loss Weight: {}".format(self.current_loss_weight))
        # print("Loss Before: {}".format(loss))
        loss = loss * self.current_loss_weight
        # loss = loss * 1
        # print("Loss After: {}".format(loss))

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label_a)[0].item()
        }
        examples_difficulty = self.compute_difficulty(img_id, label_a, label_b, output, input.grad)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary, examples_difficulty

    def parse_batch_train(self, batch):
        images = batch["img"]
        target = batch["label"]
        domain = batch["domain"]
        img_id = batch["img_id"]

        images = images.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)
        images, target_a, target_b, lam = self.domain_mix(
            images, target, domain
        )
        return images, target_a, target_b, lam, img_id

    def domain_mix(self, x, target, domain):
        lam = (
            self.dist_beta.rsample((1, ))
            if self.alpha > 0 else torch.tensor(1)
        ).to(x.device)

        # random shuffle
        perm = torch.randperm(x.size(0), dtype=torch.int64, device=x.device)
        if self.mix_type == "crossdomain":
            domain_list = torch.unique(domain)
            if len(domain_list) > 1:
                for idx in domain_list:
                    cnt_a = torch.sum(domain == idx)
                    idx_b = (domain != idx).nonzero().squeeze(-1)
                    cnt_b = idx_b.shape[0]
                    perm_b = torch.ones(cnt_b).multinomial(
                        num_samples=cnt_a, replacement=bool(cnt_a > cnt_b)
                    )
                    perm[domain == idx] = idx_b[perm_b]
        elif self.mix_type != "random":
            raise NotImplementedError(
                f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}."
            )
        mixed_x = lam*x + (1-lam) * x[perm, :]
        target_a, target_b = target, target[perm]
        return mixed_x, target_a, target_b, lam

    def compute_difficulty(self, img_id, label_a, label_b, pred, input_grad):
        alpha = 0.5

        examples_difficulty = []
        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_img_label_a = label_a[i].item()
            current_img_label_b = label_b[i].item()
            current_img_pred_conf_a = F.softmax(pred[i], dim=0).cpu().detach().numpy()[current_img_label_a]
            current_img_pred_conf_b = F.softmax(pred[i], dim=0).cpu().detach().numpy()[current_img_label_b]

            current_img_grad_magnitude = gradient_magnitude(input_grad[i].cpu().numpy(), channel=True)

            current_img_difficulty = (1 - alpha) * (current_img_grad_magnitude / current_img_pred_conf_a) + \
                                     alpha * (current_img_grad_magnitude / current_img_pred_conf_b)

            CL_example = curriculum_example(
                img_id=current_img_id,
                difficulty=current_img_difficulty
            )
            examples_difficulty.append(CL_example)

        return examples_difficulty



