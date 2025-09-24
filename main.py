import fiftyone as fo
from pipeline import run_pipeline
from prepare_data.data_cleaner import get_file_extensions


def main():
    run_pipeline()
    get_file_extensions('./data/processed/data/')
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path='./data/processed/data',
        labels_path='./data/processed/data/_annotations.coco.json',
    )
    print(dataset)
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    main()
