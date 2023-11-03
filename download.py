from functools import partial
import duckdb
import zipfile
import os
import glob
import cv2
import numpy as np
from PIL import Image
import subprocess
import shutil
from tqdm import tqdm
import fcntl
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from upload import auth_flow, upload_file, upload_all


def load_total_bvids(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    bvs = con.sql("SELECT bv FROM bilibili").to_df()["bv"].to_list()
    return bvs


def load_downloaded_bvids(downloaded_path: str):
    bvs = open(downloaded_path, "r").read().splitlines()
    return bvs


def download(bv: str, dst_dir: str, bbdown_bin: str):
    p = subprocess.run(
        f'{bbdown_bin} {bv} --work-dir {dst_dir} -F <bvid> --video-only --skip-cover -p 1 -M <bvid>/<bvid> -e "hevc,avc,av1"'.split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=True,
    )
    if p.returncode != 0:
        return None
    else:
        return bv


def count_total_frames(path: str, yield_freq: float = 0.25):
    vidcap = cv2.VideoCapture(path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    org_total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, fps / yield_freq)
    total_frames = int(org_total_frames / interval)
    return total_frames


def zipdir(root: str, dst_dir: str):
    name = os.path.basename(root)
    dst_zip_path = os.path.join(dst_dir, name + ".zip")
    with zipfile.ZipFile(dst_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_name in os.listdir(root):
            src_file = os.path.join(root, src_name)
            zf.write(src_file, src_name)
    return dst_zip_path


def remove_along_till_root(root: str, path: str):
    assert os.path.commonprefix([root, path]) == root
    while path != root:
        try:
            if os.path.isfile(path):
                os.remove(path)
            else:
                os.rmdir(path)
        except Exception as e:
            print(f"Exception from remove {path}: {e}")
        path = os.path.dirname(path)



def center_crop(image):
    height, width = image.shape[:2]
    center_width = width // 2
    offset = int(min(width, height) / 2)
    center_height = height // 2
    return image[
        center_height - offset : center_height + offset,
        center_width - offset : center_width + offset,
        :,
    ]


def frames(path: str, yield_freq: float = 0.25, image_size: int = 512):
    vidcap = cv2.VideoCapture(path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    interval = int(fps / yield_freq)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % interval == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = center_crop(image)
            image = cv2.resize(
                image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4
            )
            yield image
        count += 1


def video2frames(path: str, dst_dir: str, image_size: int = 512, verbose: bool = True):
    yield_freq = 0.25
    num_frames = count_total_frames(path, yield_freq)
    if num_frames < 8:
        return
    pbar = tqdm(total=num_frames, disable=not verbose)
    for idx, frame in enumerate(frames(path, yield_freq, image_size)):
        frame = Image.fromarray(frame)
        frame.save(os.path.join(dst_dir, f"{idx:06d}.jpg"))
        pbar.update()



def remove_along_till_root(root: str, path: str):
    assert os.path.commonprefix([root, path]) == root
    while path != root:
        try:
            if os.path.isfile(path):
                os.remove(path)
            else:
                os.rmdir(path)
        except Exception as e:
            print(f"Exception from remove {path}: {e}")
        path = os.path.dirname(path)

