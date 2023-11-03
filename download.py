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


def load_total_bvids(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    bvs = con.sql("SELECT bv FROM bilibili").to_df()["bv"].to_list()
    return bvs


def load_downloaded_bvids(downloaded_path: str):
    with open(downloaded_path) as f:
        bvs = f.readlines()
    bvs = [bv.strip() for bv in bvs]
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


def video_to_frames(video_path: str, frame_root: str):
    video_name = os.path.basename(video_path).split(".")[0]
    dst_dir = os.path.join(frame_root, video_name)
    os.makedirs(dst_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)


def count_total_frames(path: str, yield_freq: float = 0.25):
    vidcap = cv2.VideoCapture(path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    org_total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, fps / yield_freq)
    total_frames = int(org_total_frames / interval)
    return total_frames


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
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def pipeline(
    bv: str,
    downloaded_path: str = "downloaded.txt",
    bbdown_bin: str = "bin/BBDown",
    cache_dir: str = "data/videos",
    data_dir: str = "data/frames",
    image_size: int = 512,
    pbar: tqdm = None,
):
    ret = download(bv, cache_dir, bbdown_bin)
    if ret is not None:
        video_path = (
            glob.glob(os.path.join(cache_dir, f"{bv}.mp4"))
            + glob.glob(os.path.join(cache_dir, "**", f"{bv}.mp4"), recursive=True)
        )[0]
        frame_dir = os.path.join(data_dir, f"{bv}")
        os.makedirs(frame_dir, exist_ok=True)
        video2frames(  # remove the directory
            video_path,
            frame_dir,
            image_size,
            False,
        )
        with open(downloaded_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(ret)
            f.write("\n")
            fcntl.flock(f, fcntl.LOCK_UN)
        zipdir(frame_dir, data_dir)
        remove_along_till_root(cache_dir, video_path)
        shutil.rmtree(frame_dir)
    if pbar is not None:
        pbar.update()


def main(
    db_path: str = "bilibili.db",
    downloaded_path: str = "downloaded.txt",
    bbdown_bin: str = "bin/BBDown",
    cache_dir: str = "data/videos",
    data_dir: str = "data/frames",
    image_size: int = 512,
    verbose: bool = True,
):
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    bvs = load_total_bvids(db_path)
    downloaded_bvs = load_downloaded_bvids(downloaded_path)
    bvs = set(bvs) - set(downloaded_bvs)
    pbar = tqdm(total=len(bvs), disable=not verbose)
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        pool.map(
            partial(
                pipeline,
                downloaded_path=downloaded_path,
                bbdown_bin=bbdown_bin,
                cache_dir=cache_dir,
                data_dir=data_dir,
                image_size=image_size,
                pbar=pbar,
            ),
            bvs,
        )
        pool.shutdown(wait=True)
    pbar.close()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
