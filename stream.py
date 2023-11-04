import asyncio
import datetime
import httpx
import ffmpeg
from PIL import Image
import zipfile
import io
import lz4.block
import json
import sys
from aiofiles import os as aioos, open as aioopen
import os
import duckdb
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
import cv2
from cryptography.hazmat.primitives import serialization

MIXIN_KEY_TABLE = [
    46,
    47,
    18,
    2,
    53,
    8,
    23,
    32,
    15,
    50,
    10,
    31,
    58,
    3,
    45,
    35,
    27,
    43,
    5,
    49,
    33,
    9,
    42,
    19,
    29,
    28,
    14,
    39,
    12,
    38,
    41,
    13,
    37,
    48,
    7,
    16,
    24,
    55,
    40,
    61,
    26,
    17,
    0,
    1,
    60,
    51,
    30,
    4,
    22,
    25,
    54,
    21,
    56,
    59,
    6,
    63,
    57,
    62,
    11,
    36,
    20,
    34,
    44,
    52,
]


async def get_sessdata():
    api_url = "http://api.bilibili.com/x/web-interface/nav"
    cookies = {
        "SESSDATA": "84b4e0f1%2C1714364387%2Cac8ab%2Ab1CjBCYFhCKr4Xec1ne49hSSMyd22JakNTXIjBOQc-egEAO2Jr_BfbHA1IzsBiC8sM0X4SVl9ybEFOcWZCbmlYX0VrUzE1ZW8zMU9KYWItTk1LTHBOSXFieXN2VV9pdUxjX3d5NTJodjE2bjk3UUdmM0s5aHdDSFlPOUNWWmxCWWZlem1BVXU1VmtBIIEC",
        "bili_jct": "8f5bf930a0a94dce6036be93d971c723",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(api_url, cookies=cookies, follow_redirects=True)
        wbi_img = resp.json()["data"]["wbi_img"]
        img = wbi_img["img_url"].split("/")[-1].replace(".png", "")
        sub = wbi_img["sub_url"].split("/")[-1].replace(".png", "")
        key = sub + img
        mixin_key = [key[MIXIN_KEY_TABLE[i]] for i in range(64)]
        mixin_key = "".join(mixin_key)
        params = {
            "zab": mixin_key,
            "wts": str(datetime.datetime.utcnow().timestamp()).split(".")[0],
        }
        return params, cookies


async def seek_stream(bvid: str, cid: str, sessdata: dict, cookies: dict, **kwargs):
    api_url = f"https://api.bilibili.com/x/player/playurl"
    params = {
        "bvid": bvid,
        "cid": cid,
        "qn": 116,
        "fnval": 16,
        "fnver": 0,
        "fourk": 1,
        "platform": "html5",
        "high_quality": 1,
        **sessdata,
    }
    # headers = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            api_url,
            params=params,
            cookies=cookies,  # headers=headers
        )
        resp.raise_for_status()

        print(resp.json().keys())
        if "data" not in resp.json():
            return None

        segments = resp.json()["data"]["dash"]["video"]

        return segments


async def get_video_info(
    bvid: str, sessdata: dict, cookies: dict, pidx: int = 0, **kwargs
):
    api_url = f"https://api.bilibili.com/x/player/pagelist"
    params = {"bvid": bvid, "jsonp": "jsonp", **sessdata}
    async with httpx.AsyncClient() as client:
        resp = await client.get(api_url, params=params, cookies=cookies)
        resp.raise_for_status()
        if "data" not in resp.json():
            return None
        pagelists = resp.json()["data"]
        page = pagelists[pidx]

        return {
            "title": page["part"],
            "cid": page["cid"],
            "bvid": bvid,
            "duration": page["duration"],
            "width": page["dimension"]["width"],
            "height": page["dimension"]["height"],
            "first_frame": page.get("first_frame", None),
        }


async def center_crop_resize(frame: Image.Image, size: int):
    width, height = frame.size
    min_d = min(width, height)
    diff = abs(width - height)
    if width > height:
        frame = frame.crop((diff / 2, 0, diff / 2 + min_d, min_d))
    else:
        frame = frame.crop((0, diff / 2, min_d, diff / 2 + min_d))
    frame = frame.resize((size, size), Image.Resampling.LANCZOS)
    return frame


async def download_videos(
    bvids: list[str],
    data_dir: str = "data/MVFdataset/train/",
    interval: float = 5.0,
    **kwargs,
) -> list[bytes]:
    import json
    import yt_dlp

    URL = "https://www.bilibili.com/video/{bvid}"
    urls = [URL.format(bvid=bvid) for bvid in bvids]
    # Download only videos longer than a minute (or with unknown duration)
    # ℹ️ See help(yt_dlp.YoutubeDL) for a list of available options and public functions
    try:
        ydl_opts = {
            "format": "bv[ext=mp4][height<=480]",
            "outtmpl": f"{data_dir}/%(webpage_url_basename)s.%(ext)s",
            "concurrent_fragments": 16,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)
            return True
    except Exception:
        return False


async def cut_video(video_path: str, image_size: int, interval: float = 5.0):
    import cv2

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_stride = int(interval * fps)
    frames = []
    for i in range(0, frame_count, frame_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = await center_crop_resize(frame, image_size)
            frames.append(frame)
    return frames


async def cut_videos(data_dir: str, image_size: int, interval: float = 5.0):
    """
    Returns an array of frames and a dict of video metadata
    """
    frames = []
    for video_name in os.listdir(data_dir):
        bv = os.path.splitext(video_name)[0]
        video_path = os.path.join(data_dir, video_name)
        video_frames = await cut_video(video_path, image_size, interval)
        zf_path = os.path.join(data_dir, f"{bv}.zip")
        with zipfile.ZipFile(zf_path, "w") as zf:
            for i in range(len(video_frames)):
                bio = io.BytesIO()
                frame = video_frames[i]
                frame.save(bio, "JPEG")
                bio.seek(0)
                zf.writestr(f"{i:04d}.jpeg", bio.getvalue())
    return frames


async def remove_cached_video(bvids: list[str], data_dir: str = "data/MVFdataset/"):
    for bvid in bvids:
        os.remove(os.path.join(data_dir, bvid + ".mp4"))


async def capture_video(
    bvids: list[str],
    data_dir: str = "data/MVFdataset/train/",
    interval: float = 5.0,
    image_size: int = 512,
    **kwargs,
):
    await aioos.makedirs(data_dir, exist_ok=True)
    ready = await download_videos(bvids, data_dir=data_dir, interval=interval)
    if ready:
        await cut_videos(data_dir=data_dir, image_size=image_size, interval=interval)
        await remove_cached_video(bvids, data_dir=data_dir)


async def create_bilibili_table(db_path: str):
    conn = duckdb.connect(db_path)
    conn.sql("CREATE SEQUENCE IF NOT EXISTS bilibili_id START WITH 1")
    conn.sql(
        "CREATE TABLE IF NOT EXISTS bilibili (id INTEGER NOT NULL PRIMARY KEY DEFAULT NEXTVAL('bilibili_id'), bvid VARCHAR(255),  title VARCHAR(255), url VARCHAR(255), UNIQUE(bvid))"
    )
    conn.commit()
    conn.close()


async def create_done_table(db_path: str):
    conn = duckdb.connect(db_path)
    conn.sql(
        """
    CREATE TABLE IF NOT EXISTS collected (
        bvid VARCHAR(32) NOT NULL PRIMARY KEY,
        cid INT,
        signature VARCHAR(255),
        timestamp TIMESTAMP,
        length INT,
        width INT,
        height INT,
    )
    """
    )
    conn.commit()
    conn.close()


async def list_bvids(db_path: str):
    conn = duckdb.connect(db_path)
    # total = conn.sql("SELECT bv FROM bilibili").fetch_arrow_reader()
    # done = conn.sql("SELECT bvid FROM collected").fetch_arrow_reader()
    todo = conn.sql("SELECT bvid FROM bilibili EXCEPT SELECT bvid FROM collected")
    while res := todo.fetchone():
        yield res[0]


async def record_done(
    db_path: str,
    bvid: str,
    cid: int,
    signature: str,
    length: int,
    width: int,
    height: int,
    **kwargs,
):
    conn = duckdb.connect(db_path)
    conn.sql(
        f"INSERT INTO collected VALUES (?, ?, ?, ?, ?, ?, ?)",
        params=(
            bvid,
            cid,
            signature,
            datetime.datetime.now(),
            length,
            width,
            height,
        ),
    )
    conn.commit()
    conn.close()


async def rename_column(db_path: str):
    conn = duckdb.connect(db_path)
    conn.sql(
        """
    ALTER TABLE bilibili RENAME COLUMN bv TO bvid
    """
    )
    conn.commit()
    conn.close()


async def setup(db_path: str, data_dir: str):
    await rename_column(db_path)
    await create_bilibili_table(db_path)
    await create_done_table(db_path)
    await aioos.makedirs(data_dir, exist_ok=True)


async def main(
    db_path: str,
    data_dir: str = "data/MVFdataset/train/",
    username: str | None = None,
    image_size: int = 512,
    interval: float = 5.0,
    **kwargs,
):
    await setup(db_path, data_dir)
    username = username or "anonymous"
    batch_size = 4
    bvids = []
    async for bvid in list_bvids(db_path):
        bvids.append(bvid)
        if len(bvids) == batch_size:
            await capture_video(
                bvids, data_dir, interval=interval, image_size=image_size
            )
            bvids = []
    await capture_video(bvids, data_dir, interval=interval, image_size=image_size)


if __name__ == "__main__":
    asyncio.run(main("bilibili.db"))
