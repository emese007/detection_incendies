from prepare_data.data_loader import load_data
from prepare_data.data_explorer import (
    get_basic_informations,
    get_annotations_per_image,
    get_n_images_per_category,
    get_bbox_images_size,
)
from prepare_data.data_cleaner import (
    get_file_extensions,
    get_images_without_annotation,
    check_image_files,
    get_invalid_annotations,
    get_orphaned_annotations,
)


def run_pipeline():
    df, json_data = load_data('./data/raw/_annotations.coco.json')

    get_basic_informations(df, json_data)
    get_annotations_per_image(df)
    get_n_images_per_category(df)
    get_bbox_images_size(df, json_data)

    get_file_extensions('./data/raw/')
    get_images_without_annotation(json_data)
    check_image_files(df, './data/raw/')
    get_orphaned_annotations(df, json_data)
    get_invalid_annotations(df)
