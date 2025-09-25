from prepare_data.data_cleaner import get_images_without_annotation


def test_get_images_without_annotation():
    mock_json = {
        'images': [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}],
        'annotations': [{'image_id': 0}, {'image_id': 1}],
    }
    assert get_images_without_annotation(mock_json) == [2, 3]
