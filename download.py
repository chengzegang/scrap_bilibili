import duckdb
import zipfile
import os
import cv2
from PIL import Image
import subprocess
from tqdm import tqdm
import asyncio
from functools import wraps
from aiofiles import os, open
from os.path import join, basename, commonprefix, dirname


async def async_enumerate(aiterable, start=0):
    """Async version of enumerate
    :param aiterable: a async iterable (an object implementing the async iterator protocol)
    :param start: the counter start value
    :yields: a tuple of the form (counter, value) where counter is the next integer value and value is the next value from the async iterable
    """
    n = start
    async for elem in aiterable:
        yield n, elem
        n += 1


def afunc(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        coro = asyncio.to_thread(func, *args, **kwargs)
        return await coro

    return wrapper


def load_total_bvids(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    bvs = con.sql("SELECT bv FROM bilibili").to_df()["bv"].to_list()
    return bvs


async def load_downloaded_bvids(downloaded_path: str):
    bvs = []
    async with open(downloaded_path, "r") as f:
        async for line in f:
            bvs.append(line.strip())
    return bvs


def check_stdout(stdout: bytes):
    stdout = stdout.decode("utf-8")
    if "任务完成" in stdout:
        return True
    else:
        return False


async def download(bv: str, dst_dir: str, bbdown_bin: str):
    @afunc
    def run():
        p = subprocess.run(
            [
                bbdown_bin,
                bv,
                "--work-dir",
                dst_dir,
                "-F",
                '"<bvid>"',
                "--video-only",
                "--skip-cover",
                "-p",
                "1",
                "-M",
                '"<bvid>"',
                "-e",
                "hevc,avc,av1",
                "-mt",
                "false",
            ],
            bufsize=65536,
            capture_output=True,
        )
        return p

    p = await run()
    await asyncio.sleep(0.1)
    if check_stdout(p.stdout):
        return bv
    else:
        print(p.stdout.decode("utf-8"))
        return None


@afunc
def count_total(path: str, interval: float = 5.0):
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames // fps
    total_frames = int(durationInSeconds / interval)
    return total_frames


@afunc
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


async def zipdir(root: str, dst_dir: str):
    name = basename(root)
    dst_zip_path = join(dst_dir, name + ".zip")
    with zipfile.ZipFile(dst_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_name in await os.listdir(root):
            src_file = join(root, src_name)
            zf.write(src_file, src_name)
    return dst_zip_path


async def remove_along_till_root(root: str, path: str):
    assert commonprefix([root, path]) == root
    while path != root:
        try:
            if await os.path.isfile(path):
                await os.remove(path)
            else:
                await os.rmdir(path)
        except Exception as e:
            print(f"Exception from remove {path}: {e}")
        path = dirname(path)


async def frames(path: str, total: int, interval: float = 5.0, image_size: int = 512):
    if not await os.path.exists(path):
        raise FileNotFoundError(path)
    vidcap = cv2.VideoCapture(path)
    for i in range(total):
        milsec = i * interval * 1000
        vidcap.set(cv2.CAP_PROP_POS_MSEC, milsec)
        success, image = vidcap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = await center_crop(image)
        image = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4
        )
        yield image


async def video2frames(
    path: str,
    dst_dir: str,
    image_size: int = 512,
    verbose: bool = True,
    interval: float = 5.0,
):
    total = await count_total(path, interval)
    if total < 8:
        return
    pbar = tqdm(total=total, disable=not verbose)
    async for idx, frame in async_enumerate(frames(path, total, interval, image_size)):
        frame = Image.fromarray(frame)
        frame.save(join(dst_dir, f"{idx:06d}.jpg"))
        pbar.update()
