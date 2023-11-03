import datasets
import os


def many_video_frames(root: str):
    for dirname, subdirs, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".zip"):
                filepath = os.path.join(dirname, filename)
                yield filepath


def upload_to_hub(root: str):
    dataset = datasets.Dataset.from_generator(many_video_frames(root))
    dataset.push_to_hub("chengzegang/Bilibili-Many-Video-Frames", private=True)



if __name__ == '__main__':
    upload_to_hub('data/frames/')