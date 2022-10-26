import torch
from torch.nn import functional as F

from dg.optim import build_optimizer, build_lr_scheduler
from dg.utils import count_num_param
from dg.engine import TRAINER_REGISTRY, TrainerX
from dg.engine.trainer import SimpleNet
from dg.utils.tools import gradient_magnitude
from dg.utils.curriculum import curriculum_example


@TRAINER_REGISTRY.register()
class CrossGrad(TrainerX):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_f = cfg.TRAINER.CG.EPS_F
        self.eps_d = cfg.TRAINER.CG.EPS_D
        self.alpha_f = cfg.TRAINER.CG.ALPHA_F
        self.alpha_d = cfg.TRAINER.CG.ALPHA_D

    def build_model(self):
        cfg = self.cfg

        print("Building F")
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)

        print("Building D")
        self.D = SimpleNet(cfg, cfg.MODEL, self.num_source_domains)
        self.D.to(self.device)
        print("# params: {:,}".format(count_num_param(self.D)))
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        self.register_model("D", self.D, self.optim_D, self.sched_D)

    def forward_backward(self, batch):
        input, label, domain, img_id = self.parse_batch_train(batch)
        input.requires_grad = True

        # Compute domain perturbation
        loss_d = F.cross_entropy(self.D(input), domain)
        loss_d.backward()
        grad_d = torch.clamp(input.grad.data, min=-0.1, max=0.1)
        input_d = input.data + self.eps_f * grad_d


        # Compute label perturbation
        input.grad.data.zero_()
        loss_f = F.cross_entropy(self.F(input), label)
        loss_f.backward()
        grad_f = torch.clamp(input.grad.data, min=-0.1, max=0.1)
        input_f = input.data + self.eps_d * grad_f

        input = input.detach()
        input.requires_grad = True
        input_d.requires_grad = True

        # Update label net
        pred_f1 = self.F(input)
        loss_f1 = F.cross_entropy(pred_f1, label)
        pred_f2 = self.F(input_d)
        loss_f2 = F.cross_entropy(pred_f2, label)
        loss_f = (1 - self.alpha_f) * loss_f1 + self.alpha_f * loss_f2

        # print("Current Loss Weight: {}".format(self.current_loss_weight))
        # print("Loss Before: {}".format(loss_f))
        loss_f = loss_f * self.current_loss_weight
        # loss_f = loss_f * 1
        # print("Loss After: {}".format(loss_f))

        self.model_backward_and_update(loss_f, "F")

        # Update domain net
        loss_d1 = F.cross_entropy(self.D(input), domain)
        loss_d2 = F.cross_entropy(self.D(input_f), domain)
        loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
        self.model_backward_and_update(loss_d, "D")

        loss_summary = {"loss_f": loss_f.item(), "loss_d": loss_d.item()}

        examples_difficulty = self.compute_difficulty(img_id=img_id, label=label, pred_1=pred_f1, pred_2=pred_f2, input1_grad=input.grad, input2_grad=input_d.grad)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary, examples_difficulty

    def model_inference(self, input):
        return self.F(input)

    def compute_difficulty(self, img_id, label, pred_1, pred_2, input1_grad, input2_grad):
        alpha = 0.5

        examples_difficulty = []
        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_img_label = label[i].item()
            current_img_pred_conf = F.softmax(pred_1[i], dim=0).cpu().detach().numpy()[current_img_label]
            current_img_p_pred_conf = F.softmax(pred_2[i], dim=0).cpu().detach().numpy()[current_img_label]

            current_img_grad_magnitude = gradient_magnitude(input1_grad[i].cpu().numpy(), channel=True)
            current_img_p_grad_magnitude = gradient_magnitude(input2_grad[i].cpu().numpy(), channel=True)

            current_img_difficulty = (1 - alpha) * (current_img_grad_magnitude / current_img_pred_conf) + \
                                     alpha * (current_img_p_grad_magnitude / current_img_p_pred_conf)

            CL_example = curriculum_example(
                img_id=current_img_id,
                difficulty=current_img_difficulty
            )
            examples_difficulty.append(CL_example)

        return examples_difficulty
