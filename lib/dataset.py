from torch.utils.data import Dataset
import torch
import json
import os
import cv2
import imageio
import numpy as np


class NerfDataset(Dataset):
    def __init__(self, basedir, half_res, testskip, isMaster, mode):
        
        self.basedir = basedir
        self.mode = mode
        self.half_res = half_res
        self.testskip = testskip

        with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
            self.meta = json.load(fp)

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):

        frame = self.meta["frames"][item]
        fname = os.path.join(self.basedir, frame["file_path"] + ".png")
        img = imageio.imread(fname)
        pose = np.array(frame["transform_matrix"])
            
        img = (np.array(img) / 255.0).astype(np.float32)
        pose = np.array(pose).astype(np.float32)

        if self.half_res:
            img = torch.from_numpy(cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_AREA))

        pose = torch.from_numpy(pose) # transform matrix

        return img, pose

    def __len__(self):
        return len(self.meta["frames"])
