from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from ultralytics.data.converter import convert_coco


def copy_images(images_path):
    source_directory = Path(images_path)
    destination_directory = source_directory.parent.parent / 'converted' / 'images'
    destination_directory.mkdir(parents=True, exist_ok=True)

    for item in source_directory.glob('*.jpg'):
        shutil.copy(item, destination_directory / item.name)
    print('· Succesfully copied all images.')


def convert_annotations(coco_json_path):
    destination_directory = Path(coco_json_path).parent.parent / 'converted'
    if not destination_directory.is_dir():
        convert_coco(coco_json_path, str(destination_directory))
        copy_images(coco_json_path)


def split_dataset(converted_data_path):
    source_directory = Path(converted_data_path)
    destination_directory = source_directory.parent / 'dataset'

    images = list((source_directory / 'images').glob('*.jpg'))

    train, temp = train_test_split(images, test_size=0.3, random_state=1)
    val, test = train_test_split(temp, test_size=0.33, random_state=1)

    splits = {'train': train, 'val': val, 'test': test}

    for split_name, files in splits.items():
        images_dir = destination_directory / split_name / 'images'
        labels_dir = destination_directory / split_name / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in files:
            shutil.copy(image_path, images_dir / image_path.name)

            label_path = (
                source_directory
                / 'labels'
                / '_annotations.coco'
                / f'{image_path.stem}.txt'
            )
            if label_path.exists():
                shutil.copy(label_path, labels_dir / label_path.name)
    print('· Succesfully split dataset.')
