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


def auth_flow():
    if "DBX_ACCESS_TOKEN" in os.environ:
        return True
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
        return True
    except Exception as e:
        raise Exception("Error: {}".format(e))


def upload_file(local_path: str, remote_path: str, pbar: tqdm = None):
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
        oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
        app_key=os.environ["DBX_APP_KEY"],
        app_secret=os.environ["DBX_APP_SECRET"],
    )
    dbx.files_upload(
        open(local_path, "rb").read(),
        remote_path,
        mode=dropbox.files.WriteMode.overwrite,
    )
    if pbar is not None:
        pbar.update()
        pbar.set_description(f"Uploaded: {os.path.basename(local_path)}")


def list_exists(remote_root: str, extension: str = ".zip"):
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
        oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
        app_key=os.environ["DBX_APP_KEY"],
        app_secret=os.environ["DBX_APP_SECRET"],
    )
    filenames = []
    resp = dbx.files_list_folder(remote_root)
    while True:
        filenames += [entry.name for entry in resp.entries]
        if resp.has_more:
            resp = dbx.files_list_folder_continue(resp.cursor)
        else:
            break
    return filenames


def upload_all(local_root: str, remote_root: str, downloaded_path: str, num_threads=64):
    downloaded = set(open(downloaded_path, "r").read().splitlines())
    existed = set(
        os.path.splitext(os.path.basename(f))[0] for f in list_exists(remote_root)
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
                    continue
                local_path = os.path.join(local_root, f)
                remote_path = os.path.join(remote_root, f)
                futures.append(
                    executor.submit(upload_file, local_path, remote_path, pbar)
                )
        for f in as_completed(futures):
            if f.exception() is not None:
                print(f.exception())
                
        executor.shutdown(wait=True)
        pbar.close()
    # upload_file(downloaded_path, os.path.join(remote_root, "downloaded.txt"))


def download(local_path: str, remote_path: str):
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=os.environ["DBX_REFRESH_TOKEN"],
        oauth2_access_token=os.environ["DBX_ACCESS_TOKEN"],
        app_key=os.environ["DBX_APP_KEY"],
        app_secret=os.environ["DBX_APP_SECRET"],
    )
    metadata, response = dbx.files_download(remote_path)
    with open(local_path, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    auth_flow()
    upload_all(
        local_root="data/frames",
        remote_root="/MVFdataset/",
        downloaded_path="downloaded.txt",
    )
    # download(
    #    local_path="downloaded.txt",
    #    remote_path="/MVFdataset/downloaded.txt",
    # )
