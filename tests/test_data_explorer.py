import pandas as pd
from prepare_data.data_explorer import get_basic_informations


def test_get_basic_informations_invalid_coco_json_structure():
    mock_json = {
        'cat': [{'category_id': 0, 'name': 'fire', 'supercategory': 'wildfire'}],
        'img': [{'image_id': 0}],
        'ann': [{'annotation_id': 0, 'image_id': 0, 'category_id': 0}],
    }

    df_images = pd.json_normalize(mock_json['img'])
    df_annotations = pd.json_normalize(mock_json['ann'])
    df_categories = pd.json_normalize(mock_json['cat'])

    df_images_annotations = df_images.merge(df_annotations, 'inner', 'image_id')
    df = df_categories.merge(
        df_images_annotations,
        'inner',
        'category_id',
    )

    assert (
        get_basic_informations(df, mock_json) == 'Coco json file structure is invalid.'
    )
