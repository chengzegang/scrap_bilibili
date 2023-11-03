import glob
import os
import shutil


def clean(cache_dir: str, data_dir: str, downloaded_path: str):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    zip_filepaths = glob.glob(os.path.join(data_dir, "*.zip"))
    downloaded = set(open(downloaded_path, "r").read().splitlines())
    for zf in zip_filepaths:
        if os.path.splitext(os.path.basename(zf))[0] not in downloaded:
            print(f"Unfinished: {zf}")
            # os.remove(zf)


if __name__ == "__main__":
    clean(
        cache_dir="data/videos",
        data_dir="data/frames",
        downloaded_path="downloaded.txt",
    )
