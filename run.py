from functools import partial
import os
import glob
import shutil
from tqdm import tqdm
import fcntl
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from upload import auth_flow, upload_file, upload_all
from download import (
    download,
    load_downloaded_bvids,
    load_total_bvids,
    video2frames,
    zipdir,
    remove_along_till_root,
)
from clean import clean


def pipeline(
    bv: str,
    remote_root: str = "/MVFdataset/",
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
        zip_filepath = zipdir(frame_dir, data_dir)
        upload_file(
            zip_filepath, os.path.join(remote_root, os.path.basename(zip_filepath))
        )
        remove_along_till_root(cache_dir, video_path)
        shutil.rmtree(frame_dir)
    if pbar is not None:
        pbar.update()


def main(
    remote_root: str = "/MVFdataset/",
    db_path: str = "bilibili.db",
    downloaded_path: str = "downloaded.txt",
    bbdown_bin: str = "bin/BBDown",
    cache_dir: str = "data/videos",
    data_dir: str = "data/frames",
    image_size: int = 512,
    verbose: bool = True,
    num_threads: int = 16,
):
    clean(
        cache_dir,
        data_dir,
        downloaded_path,
    )
    auth_flow()
    upload_all(data_dir, remote_root, downloaded_path)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    bvs = load_total_bvids(db_path)
    downloaded_bvs = load_downloaded_bvids(downloaded_path)
    bvs = set(bvs) - set(downloaded_bvs)
    pbar = tqdm(total=len(bvs), disable=not verbose)
    with ThreadPoolExecutor(num_threads) as pool:
        pool.map(
            partial(
                pipeline,
                remote_root=remote_root,
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
