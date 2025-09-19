import pandas as pd
import json


def load_data(file):
    with open(file, 'r') as raw_json:
        json_data = json.load(raw_json)

    df_images = pd.json_normalize(json_data['images'])
    df_annotations = pd.json_normalize(json_data['annotations'])
    df_categories = pd.json_normalize(json_data['categories'])

    df_categories = df_categories.rename(columns={'id': 'category_id'})

    df_images_annotations = df_images.merge(df_annotations, 'inner', 'id')
    df_merged = df_categories.merge(
        df_images_annotations,
        'inner',
        'category_id',
    )
    return df_merged, json_data


# print(load_data('./data/raw/_annotations.coco.json'))

