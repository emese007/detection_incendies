from pathlib import Path
import shutil
from ultralytics.data.converter import convert_coco


def copy_images(images_path):
    source_directory = Path(images_path)
    destination_directory = source_directory.parent.parent / 'converted' / 'images'

    for item in source_directory.glob('*.jpg'):
        shutil.copy(item, destination_directory / item.name)
    print('· Succesfully copied all images.')


def convert_annotations(coco_json_path):
    destination_directory = Path(coco_json_path).parent.parent / 'converted'
    if not destination_directory.is_dir():
        convert_coco(coco_json_path, str(destination_directory))
        copy_images(coco_json_path)
