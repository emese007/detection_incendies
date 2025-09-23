# import matplotlib.pyplot as plt


def get_basic_informations(df, json_data):
    n_images = len(json_data['images'])
    n_annotations = len(json_data['annotations'])
    n_categories = len(json_data['categories'])
    category_names = df['category_name'].unique()
    print(f'· NUMBER OF IMAGES -> {n_images}')
    print(f'· NUMBER OF ANNOTATIONS -> {n_annotations}')
    print(f'· NUMBER OF CATEGORIES -> {n_categories}')
    print(f'· UNIQUE CATEGORY NAMES -> {category_names}')


def get_annotations_per_image(df):
    annotations_per_image = df.groupby('image_id').size()
    print(
        f'· ANNOTATIONS PER IMAGE -> MIN: {min(annotations_per_image)}, MAX: {max(annotations_per_image)}, AVG: {annotations_per_image.mean():.2f}'
    )
    # annotations_per_image.hist(bins='auto')
    # plt.title('Number of annotations per image')
    # plt.xlabel('Number of annotations')
    # plt.ylabel('Number of images')
    # plt.show()


def get_n_images_per_category(df):
    images_per_category = df.groupby('category_name')['image_id'].nunique()
    print(f'· NUMBER OF IMAGES PER CATEGORY ->\n{images_per_category}')


def get_bbox_images_size(df, json_data):
    images_width = [image['width'] for image in json_data['images']]
    images_height = [image['height'] for image in json_data['images']]
    bboxes_width = df['bbox'].apply(lambda x: x[2])
    bboxes_height = df['bbox'].apply(lambda x: x[3])
    print(
        f'· IMAGE WIDTH -> MIN: {min(images_width)}, MAX: {max(images_width)}, AVG: {sum(images_width) / len(images_width)}'
    )
    print(
        f'· IMAGE HEIGHT -> MIN: {min(images_height)}, MAX: {max(images_height)}, AVG: {sum(images_height) / len(images_height)}'
    )
    if min(images_height) == max(images_height) and min(images_width) == max(
        images_width
    ):
        print('  All images are the same size.')
    print(
        f'· BBOX WIDTH -> MIN: {bboxes_width.min():.2f}, MAX: {bboxes_width.max():.2f}, AVG: {bboxes_width.mean():.2f}'
    )
    print(
        f'· BBOX HEIGHT -> MIN: {bboxes_height.min():.2f}, MAX: {bboxes_height.max():.2f}, AVG: {bboxes_height.mean():.2f}'
    )
    print(
        f'· BBOX AREA -> MIN: {df["area"].min():.2f}, MAX: {df["area"].max():.2f}, AVG: {df["area"].mean():.2f}'
    )
    # df['area'].hist(bins='auto')
    # plt.title('Distribution of bbox areas')
    # plt.xlabel('Area (pixels)')
    # plt.ylabel('Number of bboxes')
    # plt.show()
