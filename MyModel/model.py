import torch
from lib import utils
from lib.model_interface import ModelInterface
from MyModel.loss import MyModelLoss
import os, tqdm, wandb, torchvision
import numpy as np
import cv2, time
from nerf import (get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, get_params, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)


class MyModel(ModelInterface):
    def set_networks(self):
        self.H, self.W, self.focal, self.render_poses = get_params(self.args.dataset["basedir"], half_res=self.args.dataset["half_res"])

        self.encode_position_fn = get_embedding_function(
            num_encoding_functions=self.args["models"]["coarse"]["num_encoding_fn_xyz"],
            include_input=self.args["models"]["coarse"]["include_input_xyz"],
            log_sampling=self.args["models"]["coarse"]["log_sampling_xyz"],
        )

        self.encode_direction_fn = None
        if self.args.models["coarse"]["use_viewdirs"]:
            self.encode_direction_fn = get_embedding_function(
                num_encoding_functions=self.args["models"]["coarse"]["num_encoding_fn_dir"],
                include_input=self.args["models"]["coarse"]["include_input_dir"],
                log_sampling=self.args["models"]["coarse"]["log_sampling_dir"],
            )

        # Initialize a coarse-resolution model.
        self.model_coarse = getattr(models, self.args["models"]["coarse"]["type"])(
            num_encoding_fn_xyz=self.args["models"]["coarse"]["num_encoding_fn_xyz"],
            num_encoding_fn_dir=self.args["models"]["coarse"]["num_encoding_fn_dir"],
            include_input_xyz=self.args["models"]["coarse"]["include_input_xyz"],
            include_input_dir=self.args["models"]["coarse"]["include_input_dir"],
            use_viewdirs=self.args["models"]["coarse"]["use_viewdirs"],
        )
        self.model_coarse.cuda().train()

        # If a fine-resolution model is specified, initialize it["
        self.model_fine = None
        if self.args.models.__contains__("fine"):
            self.model_fine = getattr(models, self.args["models"]["fine"]["type"])(
                num_encoding_fn_xyz=self.args["models"]["fine"]["num_encoding_fn_xyz"],
                num_encoding_fn_dir=self.args["models"]["fine"]["num_encoding_fn_dir"],
                include_input_xyz=self.args["models"]["fine"]["include_input_xyz"],
                include_input_dir=self.args["models"]["fine"]["include_input_dir"],
                use_viewdirs=self.args["models"]["fine"]["use_viewdirs"],
            )
            self.model_fine.cuda().train()


    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        # self.model_coarse = torch.nn.parallel.DistributedDataParallel(self.model_coarse, device_ids=[self.gpu]=False, find_unused_parameters=True).module
        self.model_coarse = torch.nn.parallel.DistributedDataParallel(self.model_coarse, device_ids=[self.gpu]).module
        self.model_fine = torch.nn.parallel.DistributedDataParallel(self.model_fine, device_ids=[self.gpu]).module

    def set_optimizers(self):
        trainable_parameters = list(self.model_coarse.parameters())
        if self.model_fine is not None:
            trainable_parameters += list(self.model_fine.parameters())
        self.optimizer = getattr(torch.optim, self.args["optimizer"]["type"])(trainable_parameters, lr=self.args["optimizer"]["lr"])

    def load_checkpoint(self):
        checkpoint = torch.load(self.args.load_checkpoint)
        self.model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            self.model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]
        return start_iter

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.args)

    def go_step(self, global_step):
        # load batch

        self.model_coarse.train()
        if self.model_fine:
            self.model_fine.train()

        ###########################
        # 6-1. mini batch
        ###########################

        try:
            img, pose = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            img, pose = next(self.train_iterator)

        img_target, pose = img[0].to(self.gpu), pose[0].to(self.gpu)
        pose_target = pose[:3, :4]

        if self.args.nerf["train"]["white_background"]:
            self.images = self.images[..., :3] * self.images[..., -1:] + (1.0 - self.images[..., -1:])

        ray_origins, ray_directions = get_ray_bundle(self.H, self.W, self.focal, pose_target)
        coords = torch.stack(meshgrid_xy(torch.arange(int(self.H)).cuda(), torch.arange(int(self.W)).cuda()), dim=-1).reshape((-1, 2))
        
        select_inds = np.random.choice(coords.shape[0], size=(self.args.nerf["train"]["num_random_rays"]), replace=False)
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

        # run
        rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
            self.H,
            self.W,
            self.focal,
            self.model_coarse,
            self.model_fine,
            ray_origins,
            ray_directions,
            self.args,
            mode="train",
            encode_position_fn=self.encode_position_fn,
            encode_direction_fn=self.encode_direction_fn,
        )

        loss = self.loss_collector.get_loss(rgb_coarse, rgb_fine, target_s)

        ###########################
        # 6-3. update
        ###########################

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = self.args.scheduler["lr_decay"] * 1000
        lr_new = self.args.optimizer["lr"] * (self.args.scheduler["lr_decay_factor"] ** (global_step / num_decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_new

    def do_validation(self, global_step):
        self.model_coarse.eval()
        if self.model_fine:
            self.model_fine.eval()

        print(1)
        time.sleep(5)
        start = time.time()
        with torch.no_grad():
            
            try:
                img, pose = next(self.valid_iterator)
            except StopIteration:
                self.valid_iterator = iter(self.valid_dataloader)
                img, pose = next(self.valid_iterator)

            img_target, pose = img[0].to(self.gpu), pose[0].to(self.gpu)
            pose_target = pose[:3, :4]
                
            ray_origins, ray_directions = get_ray_bundle(self.H, self.W, self.focal, pose_target)
            
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                self.H,
                self.W,
                self.focal,
                self.model_coarse,
                self.model_fine,
                ray_origins,
                ray_directions,
                self.args,
                mode="validation",
                encode_position_fn=self.encode_position_fn,
                encode_direction_fn=self.encode_direction_fn,
            )
            coarse_loss = img2mse(rgb_coarse[..., :3], img_target[..., :3]).item()
            if rgb_fine is not None:
                fine_loss = img2mse(rgb_fine[..., :3], img_target[..., :3]).item()
            loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
            psnr = mse2psnr(loss)

            self.val_loss_dict = {}
            self.val_loss_dict["loss_valid"] = loss,
            self.val_loss_dict["coarse_loss_valid"] = coarse_loss
            if rgb_fine is not None:
                self.val_loss_dict["fine_loss_valid"] = fine_loss
            self.val_loss_dict["psnr_valid"] = psnr
            
            os.makedirs(f"{self.args.save_root}/{self.args.run_id}/validation/rgb_coarse/", exist_ok=True)
            cv2.imwrite(f"{self.args.save_root}/{self.args.run_id}/validation/rgb_coarse/{str(global_step).zfill(5)}.png", self.cast_to_image(rgb_coarse[..., :3])[:, :, ::-1])
            if rgb_fine is not None:
                os.makedirs(f"{self.args.save_root}/{self.args.run_id}/validation/rgb_fine/", exist_ok=True)
                os.makedirs(f"{self.args.save_root}/{self.args.run_id}/validation/img_target/", exist_ok=True)
                cv2.imwrite(f"{self.args.save_root}/{self.args.run_id}/validation/rgb_fine/{str(global_step).zfill(5)}.png", self.cast_to_image(rgb_fine[..., :3])[:, :, ::-1])
                cv2.imwrite(f"{self.args.save_root}/{self.args.run_id}/validation/img_target/{str(global_step).zfill(5)}.png", self.cast_to_image(img_target[..., :3])[:, :, ::-1])

            tqdm.tqdm.write(f"Validation loss: {str(round(loss, 4))} | Validation PSNR: {str(round(psnr, 2))} | Time: {str(round(time.time() - start, 2))}")

        ###########################
        # 6-6. Checkpoints
        ###########################

    def save_checkpoint(self, global_step):
        os.makedirs(f"{self.args.save_root}/{self.args.run_id}/checkpoints", exist_ok=True)
        checkpoint_dict = {
            "iter": global_step,
            "model_coarse_state_dict": self.model_coarse.state_dict(),
            "model_fine_state_dict": None
            if not self.model_fine
            else self.model_fine.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint_dict, f"{self.args.save_root}/{self.args.run_id}/checkpoints/{str(global_step).zfill(5)}.ckpt")
        tqdm.tqdm.write("================== Saved Checkpoint =================")

    @property
    def loss_collector(self):
        return self._loss_collector
        
    def cast_to_image(self, tensor):
        tensor = tensor.permute(2, 0, 1)
        img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
        return img
        
        