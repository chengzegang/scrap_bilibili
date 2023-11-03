import dropbox
import dotenv
import dropbox.files
import os
import datasets
import zipfile
from PIL import Image
import torchvision.transforms.v2.functional as TF
import io
import torch

dotenv.load_dotenv()
dbx = dropbox.Dropbox(
    app_key=os.environ["DBX_APP_KEY"], app_secret=os.environ["DBX_APP_SECRET"]
)
shared_link = "https://www.dropbox.com/scl/fo/z3y2lpkse2ttkv7r6juwp/h?rlkey=h280022nxtq853sona7a44n3l&dl=0"
sl = dropbox.files.SharedLink(url=shared_link)
dbx.files_list_folder(path="", shared_link=sl)


class MVFDBXDataset(datasets.DatasetBuilder):
    def __init__(self, shared_link: str, **kwargs):
        super().__init__(**kwargs)
        self.dbx = dropbox.Dropbox(
            app_key=os.environ["DBX_APP_KEY"], app_secret=os.environ["DBX_APP_SECRET"]
        )
        self.shared_link = shared_link

    @property
    def ls(self):
        resp = dbx.files_list_folder(
            path="", shared_link=dropbox.files.SharedLink(url=shared_link)
        )
        while True:
            for entry in resp.entries:
                yield entry.path_lower
            if resp.has_more:
                resp = dbx.files_list_folder_continue(resp.cursor)
            else:
                break

    def download(self, remote_path: str):
        resp = self.dbx.sharing_get_shared_link_file(
            url=self.shared_link, path=remote_path
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
        yield {"bv": bv, "frames": imgs}

    def _split_generators(self, dl_manager):
        dl_manager.download()

    def _info(self):
        pass

    def _generate_examples(self):
        pass
