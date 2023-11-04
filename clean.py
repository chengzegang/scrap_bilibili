import glob
import os
import shutil
from aiofiles import os, open
from os.path import join, splitext, basename


async def clean(cache_dir: str, data_dir: str, downloaded_path: str):
    downloaded = set()
    async with open(downloaded_path, "r") as f:
        async for line in f:
            downloaded.add(line.strip())
    if await os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    for file in await os.listdir(data_dir):
        path = join(data_dir, file)
        if await os.path.isfile(path) and path.endswith(".zip"):
            if splitext(basename(file))[0] not in downloaded:
                print(f"Unfinished: {file}")
                await os.remove(path)
        else:
            if await os.path.isdir(path):
                print(f"Unfinished: {path}")
                shutil.rmtree(path)


if __name__ == "__main__":
    clean(
        cache_dir="data/videos",
        data_dir="data/frames",
        downloaded_path="downloaded.txt",
    )
