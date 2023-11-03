from datasets import load_dataset, Dataset
import zipfile
from PIL import Image
import io
import glob
import os
import torch
import torchvision.transforms.v2.functional as TF


class MVFdataset:
    def __init__(self, root: str):
        self.root = root

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):
        for path in glob.iglob(os.path.join(self.root, "*.zip")):
            imgs = []
            with zipfile.ZipFile(path) as zf:
                for name in sorted(zf.namelist()):
                    print
                    img_bytes = zf.read(name)
                    img = Image.open(io.BytesIO(img_bytes))
                    img = TF.pil_to_tensor(img)
                    imgs.append(img)
            imgs = torch.stack(imgs)
            yield {"frames": imgs, "bv_id": os.path.splitext(os.path.basename(path))[0]}


def upload(local_root: str):
    dataset = Dataset.from_generator(MVFdataset(local_root))
    dataset.push_to_hub("chengzegang/Bilibili-Many-Video-Frames", private=True)


upload("/home/zc2309/workspace/scrap_bilibili/data/frames")
