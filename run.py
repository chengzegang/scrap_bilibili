from functools import partial
import os
import glob
import shutil
from tqdm import tqdm
import fcntl
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from upload import auth_flow, upload_file2, upload_all
from dropbox import Dropbox
from download import (
    download,
    load_downloaded_bvids,
    load_total_bvids,
    video2frames,
    zipdir,
    remove_along_till_root,
)
from clean import clean
import asyncio
from aiofiles import os, open as aopen
from os.path import join, splitext, basename


async def pipeline(
    bv: str,
    dbx: Dropbox,
    remote_root: str = "/MVFdataset/",
    downloaded_path: str = "downloaded.txt",
    bbdown_bin: str = "bin/BBDown",
    cache_dir: str = "data/videos",
    data_dir: str = "data/frames",
    image_size: int = 512,
    pbar: tqdm = None,
):
    ret = await download(bv, cache_dir, bbdown_bin)
    if ret is None:
        pbar.write(f"Failed to download {bv}")
        pbar.update()
        return False
    else:
        pbar.write(f"Downloaded {bv}")
        video_path = join(cache_dir, f"{bv}.mp4")
        if not await os.path.exists(video_path):
            pbar.write(f"Failed to locate {bv}.mp4")
            pbar.update()
            return False

        frame_dir = join(data_dir, f"{bv}")
        await os.makedirs(frame_dir, exist_ok=True)
        await video2frames(  # remove the directory
            video_path,
            frame_dir,
            image_size,
            False,
        )
        zip_filepath = await zipdir(frame_dir, data_dir)
        with open(downloaded_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(ret)
            f.write("\n")
            fcntl.flock(f, fcntl.LOCK_UN)
        # upload_file2(
        #    dbx, zip_filepath, join(remote_root, os.path.basename(zip_filepath))
        # )
        await os.remove(video_path)
        shutil.rmtree(frame_dir)
        pbar.write(f"Finished {bv}")
    if pbar is not None:
        pbar.update()
    return True


async def main(
    remote_root: str = "/MVFdataset/",
    db_path: str = "bilibili.db",
    downloaded_path: str = "downloaded.txt",
    bbdown_bin: str = "bin/BBDown",
    cache_dir: str = "data/videos",
    data_dir: str = "data/frames",
    image_size: int = 512,
    verbose: bool = True,
):
    await clean(
        cache_dir,
        data_dir,
        downloaded_path,
    )
    dbx = auth_flow()
    upload_all(dbx, data_dir, remote_root, downloaded_path)
    await os.makedirs(cache_dir, exist_ok=True)
    await os.makedirs(data_dir, exist_ok=True)
    bvs = load_total_bvids(db_path)
    downloaded_bvs = await load_downloaded_bvids(downloaded_path)
    bvs = set(bvs) - set(downloaded_bvs)
    bvs = list(bvs)
    pbar = tqdm(total=len(bvs), disable=not verbose, dynamic_ncols=True)
    func = partial(
        pipeline,
        dbx=dbx,
        remote_root=remote_root,
        downloaded_path=downloaded_path,
        bbdown_bin=bbdown_bin,
        cache_dir=cache_dir,
        data_dir=data_dir,
        image_size=image_size,
        pbar=pbar,
    )
    batch_size = mp.cpu_count() * 2
    for i in range(0, len(bvs), batch_size):
        await asyncio.gather(*[func(bv) for bv in bvs[i : i + batch_size]])


if __name__ == "__main__":
    asyncio.run(main())
