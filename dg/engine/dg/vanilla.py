from torch.nn import functional as F

from dg.engine import TRAINER_REGISTRY, TrainerX
from dg.metrics import compute_accuracy
from dg.utils.tools import gradient_magnitude
from dg.utils.curriculum import curriculum_example


@TRAINER_REGISTRY.register()
class Vanilla(TrainerX):
    """Vanilla baseline."""

    def forward_backward(self, batch):
        input, label, img_id = self.parse_batch_train(batch)
        input.requires_grad = True

        pred = self.model(input)
        loss = F.cross_entropy(pred, label)

        # print("Current Loss Weight: {}".format(self.current_loss_weight))
        # print("Loss Before: {}".format(loss))
        loss = loss * self.current_loss_weight
        # loss = loss * 1
        # print("Loss After: {}".format(loss))



        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(pred, label)[0].item()
        }

        examples_difficulty = self.compute_difficulty(img_id=img_id, label=label, pred=pred, input_grad=input.grad)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary, examples_difficulty

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        img_id = batch["img_id"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, img_id

    def compute_difficulty(self, img_id, label, pred, input_grad):
        examples_difficulty = []

        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_img_label = label[i].item()
            current_img_pred_confidence = F.softmax(pred[i], dim=0).cpu().detach().numpy()[current_img_label]
            current_img_grad_magnitude = gradient_magnitude(input_grad[i].cpu().numpy(), channel=True)
            current_img_difficulty = current_img_grad_magnitude / current_img_pred_confidence

            CL_example = curriculum_example(
                img_id=current_img_id,
                difficulty=current_img_difficulty
            )
            examples_difficulty.append(CL_example)

        return examples_difficulty
