import os
import json
import shutil
import pandas as pd
from pathlib import Path


def get_file_extensions(path: str) -> set:
    """
    Get all the extensions from the files in a specified directory.
    Args:
        path (str)
    Returns:
        set
    """
    directory_path = Path(path)

    if not directory_path.is_dir():
        print(f'Error -> {path} is not a valid directory.')
        return set()

    extensions = set()

    for item in directory_path.iterdir():
        if item.is_file() and item.suffix:
            extensions.add(item.suffix.lower())

    print(f'· Found files with the following extension(s) in {path} -> {extensions}.')
    return extensions


def get_images_without_annotation(json_data: dict, need_print: bool = True) -> list:
    """
    Returns a list of image ids that do not have any annotation.
    Args:
        json_data: dict
    Returns:
        images_without_annotation: list
    """
    images_with_annotation = []
    for i in json_data['annotations']:
        if i['image_id'] not in images_with_annotation:
            images_with_annotation.append(i['image_id'])
    images_without_annotation = [
        i['id'] for i in json_data['images'] if i['id'] not in images_with_annotation
    ]
    if need_print:
        print(f'· Found {len(images_without_annotation)} images without annotations.')
    return images_without_annotation


def check_image_files(
    df: pd.DataFrame, directory_path: str, need_print: bool = True
) -> pd.DataFrame:
    """
    Checks if all images in provided dataframe actually exist in provided path.
    Args:
        df: pd.DataFrame
        directory_path: str
    Returns:
        not_found_in_directory: pd.DataFrame
    """
    df['full_path'] = df['file_name'].apply(lambda x: os.path.join(directory_path, x))
    df['file_exists'] = df['full_path'].apply(lambda x: os.path.exists(x))
    not_found_in_directory = df[~df['file_exists']]
    if need_print:
        print(
            f'· {len(not_found_in_directory)} of the images in provided dataframe were not found in {directory_path}'
        )
    return pd.DataFrame(not_found_in_directory)


def get_orphaned_annotations(
    df: pd.DataFrame, json_data: dict, need_print: bool = True
) -> pd.DataFrame:
    """
    Find all orphaned annotations.
    Args:
        df: pd.Dataframe
    Returns:
        orphaned: pd.DataFrame
    """
    valid_image_ids = [item['id'] for item in json_data['images']]

    is_valid_condition = df['image_id'].isin(valid_image_ids)
    is_orphan_condition = ~is_valid_condition

    orphaned = df[is_orphan_condition]

    if need_print:
        print(f'· Found {len(orphaned)} orphaned annotation(s).')
    return pd.DataFrame(orphaned)


def get_invalid_annotations(df: pd.DataFrame, need_print: bool = True) -> pd.DataFrame:
    """
    Find all annotations that have an invalid width and/or height or that are out of bounds.
    Args:
        df: pd.DataFrame
    Returns:
        invalid_annotations: pd.DataFrame
    """
    invalid_annotations = df[
        (df['bbox'].apply(lambda x: x[0] < 0 or x[1] < 0 or x[2] <= 0 or x[3] <= 0))
        | (df.apply(lambda row: row['bbox'][0] + row['bbox'][2] > row['width'], axis=1))
        | (
            df.apply(
                lambda row: row['bbox'][1] + row['bbox'][3] > row['height'], axis=1
            )
        )
    ]
    if need_print:
        print(
            f'· Found {len(invalid_annotations)} annotation(s) with invalid size and/or placement.'
        )
    return pd.DataFrame(invalid_annotations)


def process_data(
    directory_path: str, df: pd.DataFrame, original_json_data: dict
) -> None:
    """
    Process data by removing images that do not have annotations, images that do not exist as files, orphaned annotations, invalid annotations.
    Args:
        directory_path: str
    Returns:
        None
    """
    source_directory = Path(directory_path)

    if not source_directory.is_dir():
        print(f'Error -> {directory_path} is not a valid directory.')
        return

    destination_directory = source_directory.parent / 'processed' / 'data'

    if not os.path.exists(destination_directory):
        shutil.copytree(source_directory, destination_directory)
        print(
            f'· Succesfully copied `{source_directory}` to `{destination_directory}`.'
        )
    else:
        print(
            f'Error -> `{destination_directory}` already exists, could not process data.'
        )
        return

    path_new_json_file = destination_directory / '_annotations.coco.json'

    with open(path_new_json_file, 'r') as new_json_file:
        new_json_data = json.load(new_json_file)

    images_without_annotations_list = get_images_without_annotation(
        new_json_data, need_print=False
    )

    if images_without_annotations_list:
        new_json_data['images'] = [
            i
            for i in new_json_data['images']
            if i['id'] not in images_without_annotations_list
        ]
        print(
            f'· Succesfully deleted {len(images_without_annotations_list)} images without annotations from processed coco json file at `{path_new_json_file}`.'
        )

    invalid_annotations_df = get_invalid_annotations(df, need_print=False)

    if not invalid_annotations_df.empty:
        new_json_data['images'] = [
            i
            for i in new_json_data['images']
            if i['id'] not in invalid_annotations_df['image_id'].values
        ]
        print(
            f'· Succesfully deleted {len(invalid_annotations_df)} images that had invalid annotations from processed coco json file.'
        )
    else:
        print(
            '· No invalid annotation was found therefore no image has been deleted from processed coco json file.'
        )

    orphaned_annotations_df = get_orphaned_annotations(
        df, original_json_data, need_print=False
    )

    if not orphaned_annotations_df.empty:
        new_json_data['annotations'] = [
            i
            for i in new_json_data['annotations']
            if i['image_id'] not in orphaned_annotations_df['image_id'].values
        ]
        print(
            f'· Succesfully deleted {len(orphaned_annotations_df)} orphaned annotations from processed coco json file.'
        )
    else:
        print(
            '· No orphaned annotation was found therefore none was deleted from processed coco json file.'
        )

    images_not_found_in_directory_df = check_image_files(
        df, directory_path=str(destination_directory), need_print=False
    )

    if not images_not_found_in_directory_df.empty:
        new_json_data['images'] = [
            i
            for i in new_json_data['images']
            if i['id'] not in images_not_found_in_directory_df['image_id'].values
        ]
        print(
            f'· Succesfully deleted {len(images_not_found_in_directory_df)} images that were not found in expected directory from processed coco json file.'
        )
        new_json_data['annotations'] = [
            i
            for i in new_json_data['annotations']
            if i['image_id'] not in images_not_found_in_directory_df['image_id'].values
        ]
        print(
            f'· Succesfully deleted {len(images_not_found_in_directory_df)} annotations linked to images that were not found in expected directory from processed coco json file.'
        )
    else:
        print(
            '· All images were found in expected directory therefore nothing was deleted from processed coco json file.'
        )

    with open(path_new_json_file, 'w') as processed_json_file:
        json.dump(new_json_data, processed_json_file, indent=2)

    return None
