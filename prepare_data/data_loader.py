import pandas as pd
import json


def load_data(file):
    with open(file, "r") as raw_json:
        data = json.load(raw_json)

    images = []
    for image in data["images"]:
        images.append(image)

    df = pd.json_normalize(images)
    return df


print(load_data("data/raw/_annotations.coco.json"))
