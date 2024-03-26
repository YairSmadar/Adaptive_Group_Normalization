import os
import sys

# Add the project's root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from yolov5.utils.general import download, Path
import zipfile
import os

# Configuration
yaml = {'path': '/dev/shm/data/coco'}

# Initialize dataset root directory
dir = Path(yaml['path'])
image_dir = dir / 'images'

# Download URLs
url_base = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
label_urls = [
    url_base + 'coco2017labels.zip',  # Box labels
    url_base + 'coco2017labels-segments.zip'  # Segment labels
]
data_urls = [
    'http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
    'http://images.cocodataset.org/zips/val2017.zip'  # 1G, 5k images
]

# Function to download and unzip
def download_and_unzip(urls, destination_dir):
    downloaded_files = download(urls, dir=destination_dir, threads=3)
    for file in downloaded_files:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        os.remove(file)  # Optionally, delete the zip file after extraction

# Download and unzip labels
download_and_unzip(label_urls, dir.parent)

# Download and unzip images
download_and_unzip(data_urls, image_dir)
