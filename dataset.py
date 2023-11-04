from typing import Any
from torch.utils.data import (
    MapDataPipe,
    Dataset,
    IterableDataset,
    IterDataPipe,
    DataChunk,
    functional_datapipe,
)
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
from upload import list_exists
import torch.utils.data.datapipes as dp


@functional_datapipe("mvf_path")
class _MVFPathPipe(IterDataPipe, IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dbx = dropbox.Dropbox(
            oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
            oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
            app_key=os.environ["DBX_APP_KEY"],
            app_secret=os.environ["DBX_APP_SECRET"],
        )
        self.filenames = np.asarray(list_exists(self.dbx, "/MVFdataset", ".zip"))

    def __getitem__(self, index):
        return self.filenames[index]

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        for filename in self.filenames:
            yield filename


@functional_datapipe("dbx_download")
class _DBXDownloader(IterDataPipe, IterableDataset):
    def __init__(self, datapipe: IterDataPipe, **kwargs):
        super().__init__()
        self.datapipe = datapipe
        self.dbx = dropbox.Dropbox(
            oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
            oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
            app_key=os.environ["DBX_APP_KEY"],
            app_secret=os.environ["DBX_APP_SECRET"],
        )

    def __iter__(self):
        return len(self.datapipe)

    def download(self, remote_path: str):
        bv = os.path.basename(remote_path).split(".")[0]
        meta, resp = self.dbx.files_download(os.path.join("/MVFdataset", remote_path))
        imgs = []
        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".jpg"):
                    img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
                    img = TF.pil_to_tensor(img)
                    imgs.append(img)
        imgs = torch.stack(imgs)
        return {"bv": bv, "frames": imgs}

    def __iter__(self):
        for path in self.datapipe:
            yield self.download(path)

    def __getitem__(self, index) -> Any:
        return NotImplementedError


@functional_datapipe("mvf_dataset")
class MVFDataset(IterDataPipe, IterableDataset):
    def __init__(self, image_size: int, transform=None, **kwargs):
        super().__init__()
        self._mvfpaths = _MVFPathPipe(**kwargs)
        self.image_size = image_size
        self.transform = transform
        self.dp = (
            dp.iter.IterableWrapper(self._mvfpaths)
            .sharding_filter()
            .shuffle()
            .dbx_download()
            .map(self.resize)
            .map(transform if transform is not None else lambda x: x)
        )

    def resize(self, image: Tensor) -> Tensor:
        image["frames"] = TF.resize(image["frames"], self.image_size, antialias=True)
        return image

    def __len__(self):
        return len(self._mvfpaths)

    def __iter__(self):
        for data in self.dp:
            yield data

    def __getitem__(self, index) -> Any:
        return NotImplementedError
