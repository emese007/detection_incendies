from pathlib import Path
from ultralytics.data.converter import convert_coco


def convert_annotations(coco_json_path):
    output = Path(coco_json_path).parent.parent / 'converted'
    convert_coco(coco_json_path, str(output))
