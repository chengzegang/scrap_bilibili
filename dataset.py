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
import dropbox


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


class DBXdataset(MapDataPipe, Dataset):
    def __init__(self, shared_link: str, **kwargs):
        super().__init__(**kwargs)
        self.dbx = dropbox.Dropbox(
            app_key=os.environ["DBX_APP_KEY"], app_secret=os.environ["DBX_APP_SECRET"]
        )
        self.shared_link = shared_link
        self.filenames = np.asarray(list(self.ls))

    @property
    def ls(self):
        resp = self.dbx.files_list_folder(
            path="", shared_link=dropbox.files.SharedLink(url=self.shared_link)
        )
        while True:
            for entry in resp.entries:
                yield entry.name
            if resp.has_more:
                resp = self.dbx.files_list_folder_continue(resp.cursor)
            else:
                break

    def download(self, remote_path: str):
        resp = dropbox.Dropbox.sharing_get_shared_link_file(
            url=self.shared_link, path="/" + remote_path
        )
        imgs = []
        basename = os.path.basename(remote_path)
        bv = os.path.splitext(basename)[0]
        with zipfile.ZipFile(resp.content, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.startswith(".jpg"):
                    img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
                    img = TF.pil_to_tensor(img)
                    imgs.append(img)
        imgs = torch.stack(imgs)
        return {"bv": bv, "frames": imgs}

    def __getitem__(self, index: int) -> Tensor:
        path = self.filenames[index]
        return self.download(path)
