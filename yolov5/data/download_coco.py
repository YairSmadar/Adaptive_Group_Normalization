from utils.general import download, Path

# Configuration (You might need to adjust the 'yaml' dictionary to fit your script's requirements)
yaml = {'path': '/dev/shm/data/coco'}

# Download labels
segments = False  # segment or box labels
dir = Path(yaml['path'])  # dataset root dir
url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
download(urls, dir=dir.parent)

# Download data
urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
        'http://images.cocodataset.org/zips/val2017.zip']  # 1G, 5k images
download(urls, dir=dir / 'images', threads=3)