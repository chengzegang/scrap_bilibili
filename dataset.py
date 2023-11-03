from torch.utils.data import MapDataPipe, Dataset, IterableDataset, IterDataPipe
import datasets
import zipfile
import glob
import numpy as np
import os
from torch import Tensor
import torch
from PIL import Image
import torchvision.transforms.v2.functional as TF
import io
import webdataset as wds


class MVFDataset(MapDataPipe, Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        paths = glob.glob(os.path.join(root, "*.zip"))
        self.paths = np.asarray(paths)

    def __len__(self):
        return len(self.paths)

    def load_data(self, path: str) -> Tensor:
        imgs = []
        with zipfile.ZipFile(path, "r") as f:
            for name in sorted(f.namelist()):
                img_bytes = f.read(name)
                img = Image.open(io.BytesIO(img_bytes))
                img = TF.pil_to_tensor(img)
                imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        imgs = self.load_data(path)
        return imgs


class DBXdataset(IterDataPipe, IterableDataset):
    def __init__(self, remote_folder_link: str, cache_folder: str):
        super().__init__()
        import dropbox
        import dropbox.files

        self.remote_folder_link = remote_folder_link
        self.cache_folder = cache_folder
        self.dbx = dropbox.Dropbox()