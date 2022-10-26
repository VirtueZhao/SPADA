import torch
from torch.nn import functional as F

from dg.optim import build_optimizer, build_lr_scheduler
from dg.utils import count_num_param
from dg.engine import TRAINER_REGISTRY, TrainerX
from dg.modeling import build_network
from dg.engine.trainer import SimpleNet
from dg.utils.tools import gradient_magnitude
from dg.utils.curriculum import curriculum_example

@TRAINER_REGISTRY.register()
class DDAIG(TrainerX):
    """Deep Domain-Adversarial Image Generation.

    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        print("+Calling: DDAIG.__init__()")
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.DDAIG.LMDA
        self.clamp = cfg.TRAINER.DDAIG.CLAMP
        self.clamp_min = cfg.TRAINER.DDAIG.CLAMP_MIN
        self.clamp_max = cfg.TRAINER.DDAIG.CLAMP_MAX
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.alpha = cfg.TRAINER.DDAIG.ALPHA
        print("-Closing: DDAIG.__init__()")

    def build_model(self):
        print()
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model()")
        cfg = self.cfg
        # print(cfg)
        print("+Calling: Building Network F -> Label Classifier")
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        self.F.to(self.device)
        # print(self.F)
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().count_num_param(self.F) # Params: {:,}".format(count_num_param(self.F)))
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().count_num_param(self.F)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_optimizer(self.F, cfg.OPTIM)")
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        # print(self.optim_F)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_optimizer(self.F, cfg.OPTIM)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_lr_scheduler(self.optim_F, cfg.OPTIM)")
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        # print(self.sched_F.state_dict())
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_lr_scheduler(self.optim_F, cfg.OPTIM)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().TrainerBase.register_model(F, self.F, self.optim_F, self.sched_F)")
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().TrainerBase.register_model(F, self.F, self.optim_F, self.sched_F)")
        print("-Closing: Building Network F -> Label Classifier")
        print()

        print("+Calling: Building Network D -> Domain Classifier")
        self.D = SimpleNet(cfg, cfg.MODEL, self.num_source_domains)
        self.D.to(self.device)
        # print(self.D)
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().count_num_param(self.D) # Params: {:,}".format(count_num_param(self.D)))
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().count_num_param(self.D)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_optimizer(self.D, cfg.OPTIM)")
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        # print(self.optim_D)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_optimizer(self.D, cfg.OPTIM)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_lr_scheduler(self.optim_D, cfg.OPTIM)")
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        # print(self.sched_D.state_dict())
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_lr_scheduler(self.optim_D, cfg.OPTIM)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().TrainerBase.register_model(D, self.D, self.optim_D, self.sched_D)")
        self.register_model("D", self.D, self.optim_D, self.sched_D)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().TrainerBase.register_model(D, self.D, self.optim_D, self.sched_D)")
        print("-Closing: Building Network D -> Domain Classifier")
        print()
        print("+Calling: Building Network G -> Data Transformation Network")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_network()")
        self.G = build_network(cfg.TRAINER.DDAIG.G_ARCH, verbose=cfg.VERBOSE)
        self.G.to(self.device)
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().count_num_param(self.G) # Params: {:,}".format(count_num_param(self.G)))
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().count_num_param(self.G)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_optimizer(self.G, cfg.OPTIM)")
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        # print(self.optim_G)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_optimizer(self.G, cfg.OPTIM)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_lr_scheduler(self.optim_G, cfg.OPTIM)")
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        # print(self.sched_G.state_dict())
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().build_lr_scheduler(self.optim_G, cfg.OPTIM)")
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().TrainerBase.register_model(G, self.G, self.optim_G, self.sched_G)")
        self.register_model("G", self.G, self.optim_G, self.sched_G)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model().TrainerBase.register_model(G, self.G, self.optim_G, self.sched_G)")
        print("-Closing: Building Network G -> Data Transformation Network")
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().DDAIG.build_model()")

    def forward_backward(self, batch):
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward()")
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().parse_batch_train()")
        input, label, domain, img_id = self.parse_batch_train(batch)
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().parse_batch_train()")

        #############
        # Update G
        #############
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G")
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - generate_perturbation")
        input_p = self.G(input, lmda=self.lmda)
        # print(input_p)
        if self.clamp:
            input_p = torch.clamp(input_p, min=self.clamp_min, max=self.clamp_max)
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - generate_perturbation")
        # print()
        loss_g = 0
        # Minimize label loss
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - get_loss_from_label_classifier")
        loss_g += F.cross_entropy(self.F(input_p), label)
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - get_loss_from_label_classifier")
        # print()
        # print("F.cross_entropy(self.F(input_p), label) :{}".format(F.cross_entropy(self.F(input_p), label)))
        # print("F.cross_entropy(self.D(input_p), domain):{}".format(F.cross_entropy(self.D(input_p), domain)))
        # # Maximize domain loss
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - get_loss_from_domain_classifier")
        loss_g -= F.cross_entropy(self.D(input_p), domain)
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - get_loss_from_domain_classifier")
        # print()

        # Update Generator
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G.model_backward_and_update(loss_g, G)")
        self.model_backward_and_update(loss_g, "G")
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G.model_backward_and_update(loss_g, G)")
        # print()

        # Perturb data with new G
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - perturb data with new G")
        with torch.no_grad():
            input_p = self.G(input, lmda=self.lmda)
            if self.clamp:
                input_p = torch.clamp(input_p, min=self.clamp_min, max=self.clamp_max)
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G - perturb data with new G")
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_G")

        #############
        # Update F
        #############
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_F")
        input.requires_grad = True
        input_p.requires_grad = True

        pred = self.F(input)
        loss_f = F.cross_entropy(pred, label)

        if (self.epoch + 1) > self.warmup:
            pred_fp = self.F(input_p)
            loss_fp = F.cross_entropy(pred_fp, label)
            loss_f = (1.0 - self.alpha) * loss_f + self.alpha * loss_fp

            # print("Current Loss Weight: {}".format(self.current_loss_weight))
            # print("Loss Before: {}".format(loss_f))
            loss_f = loss_f * self.current_loss_weight
            # loss_f = loss_f * 1
            # print("Loss After: {}".format(loss_f))


        self.model_backward_and_update(loss_f, "F")

        if (self.epoch + 1) > self.warmup:
            examples_difficulty = self.compute_difficulty(img_id=img_id, label=label, pred=pred, pred_fp=pred_fp, input_grad = input.grad, input_p_grad=input_p.grad)
        else:
            examples_difficulty = None
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_F")

        #############
        # Update D
        #############
        # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_D")
        loss_d = F.cross_entropy(self.D(input), domain)
        self.model_backward_and_update(loss_d, "D")
        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_D")

        loss_summary = {
            "loss_g": loss_g.item(),
            "loss_f": loss_f.item(),
            "loss_d": loss_d.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            # print("+Calling: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_lr()")
            self.update_lr()
            # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward().update_lr()")

        # print("-Closing: train.trainer.train().SimpleTrainer.train().TrainerBase.train().TrainerX.run_epoch().DDAIG.forward_backward()")

        return loss_summary, examples_difficulty

    def model_inference(self, input):
        return self.F(input)

    def compute_difficulty(self, img_id, label, pred, pred_fp, input_grad, input_p_grad):
        examples_difficulty = []
        alpha = 0.5

        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_img_label = label[i].item()
            current_img_pred_conf = F.softmax(pred[i], dim=0).cpu().detach().numpy()[current_img_label]
            current_img_pred_p_conf = F.softmax(pred_fp[i], dim=0).cpu().detach().numpy()[current_img_label]

            current_img_grad_magnitude = gradient_magnitude(input_grad[i].cpu().numpy())
            current_img_p_grad_magnitude = gradient_magnitude(input_p_grad[i].cpu().numpy())

            current_img_difficulty = (1 - alpha) * (current_img_grad_magnitude / current_img_pred_p_conf) + \
                                     alpha * (current_img_p_grad_magnitude / current_img_pred_p_conf)

            CL_example = curriculum_example(
                img_id=current_img_id,
                difficulty=current_img_difficulty
            )
            examples_difficulty.append(CL_example)

        return examples_difficulty