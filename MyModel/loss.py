from lib.loss_interface import Loss, LossInterface
import torch.nn.functional as F
import torch
from nerf import mse2psnr


class MyModelLoss(LossInterface):
    def get_loss(self, rgb_coarse, rgb_fine, target_s):

        coarse_loss = Loss.get_L2_loss(rgb_coarse[..., :3], target_s[..., :3])
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = Loss.get_L2_loss(rgb_fine[..., :3], target_s[..., :3])
            
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)

        self.loss_dict["loss_train"] = round(loss.item(), 4)
        self.loss_dict["coarse_loss_train"] = round(coarse_loss.item(),4)
        if rgb_fine is not None:
            self.loss_dict["fine_loss_train"] = round(fine_loss.item(),4)
        self.loss_dict["psnr_train"] = round(mse2psnr(loss.item()), 2)

        return loss
        