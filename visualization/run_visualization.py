import fiftyone as fo


def run_visualization():
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path='./data/preprocessed/data',
        labels_path='./data/preprocessed/data/_annotations.coco.json',
    )
    print(dataset)
    session = fo.launch_app(dataset)
    session.wait()
