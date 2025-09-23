import os
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
        print(f'Error: {path} is not a valid directory.')
        return set()

    extensions = set()

    for item in directory_path.iterdir():
        if item.is_file() and item.suffix:
            extensions.add(item.suffix.lower())

    print(f'\nFound files with the following extension(s) in {path} -> {extensions}.')
    return extensions


def get_images_without_annotation(json_data: dict) -> list:
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
    print(f'\nFound {len(images_without_annotation)} images without annotations.')
    return images_without_annotation


def check_image_files(df: pd.DataFrame, directory_path: str) -> pd.DataFrame:
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
    print(
        f'\n{len(not_found_in_directory)} of the images in provided dataframe were not found in {directory_path}'
    )
    return pd.DataFrame(not_found_in_directory)


def get_orphaned_annotations(df: pd.DataFrame, json_data):
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

    print(f'\nFound {len(orphaned)} orphaned annotation(s).')
    return orphaned


def get_invalid_annotations(df: pd.DataFrame) -> pd.DataFrame:
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
    print(
        f'\nFound {len(invalid_annotations)} annotation(s) with invalid size and/or placement.'
    )
    return pd.DataFrame(invalid_annotations)
