import dropbox
import filelock
import dotenv
import os
import sys
import tempfile
from tqdm.auto import tqdm
import dropbox.files
from concurrent.futures import ThreadPoolExecutor, as_completed

dotenv.load_dotenv()


def auth_flow() -> dropbox.Dropbox:
    if "DBX_ACCESS_TOKEN" in os.environ:
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
            oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
            app_key=os.environ["DBX_APP_KEY"],
            app_secret=os.environ["DBX_APP_SECRET"],
        )
        return dbx
    auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        os.environ["DBX_APP_KEY"], os.environ["DBX_APP_SECRET"]
    )
    authorize_url = auth_flow.start()
    sys.stdout.write("1. Go to: {}\n".format(authorize_url))
    sys.stdout.write('2. Click "Allow" (you might have to log in first).\n')
    sys.stdout.write("3. Copy the authorization code.\n")
    auth_code = input("Enter the authorization code here: ").strip()
    try:
        oauth_result = auth_flow.finish(auth_code)
        os.environ["DBX_ACCESS_TOKEN"] = oauth_result.access_token
        os.environ["DBX_REFRESH_TOKEN"] = oauth_result.refresh_token
        with open(".env", "a") as f:
            f.write(
                f"DBX_ACCESS_TOKEN={oauth_result.access_token}\nDBX_REFRESH_TOKEN={oauth_result.refresh_token}"
            )
    except Exception as e:
        raise Exception("Error: {}".format(e))
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
        oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
        app_key=os.environ["DBX_APP_KEY"],
        app_secret=os.environ["DBX_APP_SECRET"],
    )
    return dbx


def upload_file(
    dbx: dropbox.Dropbox, local_path: str, remote_path: str, pbar: tqdm = None
):
    dbx.check_and_refresh_access_token()
    dbx.files_upload(
        open(local_path, "rb").read(),
        remote_path,
        mode=dropbox.files.WriteMode.overwrite,
    )
    if pbar is not None:
        pbar.update()
        pbar.set_description(f"Uploaded: {os.path.basename(local_path)}")


def list_exists(dbx: dropbox.Dropbox, remote_root: str, extension: str = ".zip"):
    filenames = []
    resp = dbx.files_list_folder(remote_root)
    while True:
        filenames += [entry.name for entry in resp.entries]
        if resp.has_more:
            resp = dbx.files_list_folder_continue(resp.cursor)
        else:
            break
    return filenames


def upload_file2(
    dbx: dropbox.Dropbox,
    local_path: str,
    remote_path: str,
    chunk_size: int = 4 * 1024 * 1024,
    pbar: tqdm = None,
):
    dbx.check_and_refresh_access_token()
    with open(local_path, "rb") as f:
        filesize = os.path.getsize(local_path)
        chunk_size = min(chunk_size, filesize)
        upload_session_start_result = dbx.files_upload_session_start(f.read(chunk_size))
        cursor = dropbox.files.UploadSessionCursor(
            session_id=upload_session_start_result.session_id,
            offset=f.tell(),
        )
        commit = dropbox.files.CommitInfo(path=remote_path)
        while f.tell() < filesize:
            if (filesize - f.tell()) <= chunk_size:
                res = dbx.files_upload_session_finish(
                    f.read(chunk_size), cursor, commit
                )
                print(f"Uploaded {os.path.basename(local_path)}")
            else:
                dbx.files_upload_session_append(
                    f.read(chunk_size), cursor.session_id, cursor.offset
                )
                cursor.offset = f.tell()

        if pbar is not None:
            pbar.update()
            pbar.set_description(f"Uploaded: {os.path.basename(local_path)}")
    return True


def upload_all(
    dbx: dropbox.Dropbox,
    local_root: str,
    remote_root: str,
    downloaded_path: str,
    num_threads=16,
):
    dbx.check_and_refresh_access_token()
    downloaded = set(open(downloaded_path, "r").read().splitlines())
    existed = set(
        os.path.splitext(os.path.basename(f))[0] for f in list_exists(dbx, remote_root)
    )
    with ThreadPoolExecutor(num_threads) as executor:
        files = os.listdir(local_root)
        pbar = tqdm(total=len(files))
        futures = []
        for f in files:
            if f.lower().endswith(".zip"):
                fname = os.path.splitext(f)[0]
                if fname not in downloaded or fname in existed:
                    pbar.set_description(f"{f} exists.")
                    pbar.update()
                else:
                    local_path = os.path.join(local_root, f)
                    remote_path = os.path.join(remote_root, f)
                    futures.append(
                        executor.submit(
                            upload_file2, dbx, local_path, remote_path, pbar
                        )
                    )
            else:
                pbar.set_description(f"{f} not a zip file.")
            pbar.update()
        for f in as_completed(futures):
            if f.exception() is not None:
                print(f.exception())

        executor.shutdown(wait=True)
        pbar.close()


def download(dbx: dropbox.Dropbox, local_path: str, remote_path: str):
    metadata, response = dbx.files_download(remote_path)
    with open(local_path, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    dbx = auth_flow()
    upload_all(
        dbx,
        local_root="data/frames",
        remote_root="/MVFdataset/",
        downloaded_path="downloaded.txt",
        num_threads=16,
    )
