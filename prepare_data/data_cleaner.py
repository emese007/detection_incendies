from pathlib import Path
import pandas as pd

DEFAULT_IMG_EXTS = {".jpg", ".jpeg"}


def list_extensions(image_dir: str | Path) -> set[str]:
    image_dir = Path(image_dir)
    return {p.suffix.lower() for p in image_dir.glob("*") if p.is_file()}


def check_disk_vs_coco(
    df_images: pd.DataFrame,
    image_dir: str | Path,
    valid_ext: set[str] = DEFAULT_IMG_EXTS,
    case_insensitive: bool = True,
):
    image_dir = Path(image_dir)

    def norm(name: str) -> str:
        name = Path(name).name
        return name.lower() if case_insensitive else name

    # fichiers attendus par COCO (filtrés par extension image)
    coco_files = {
        norm(fn)
        for fn in df_images["file_name"].astype(str)
        if Path(fn).suffix.lower() in valid_ext
    }

    # fichiers présents sur disque (filtrés par extension image)
    disk_files = {
        norm(p.name)
        for p in image_dir.glob("*")
        if p.is_file() and p.suffix.lower() in valid_ext
    }

    missing_on_disk = sorted(coco_files - disk_files)
    extra_on_disk = sorted(disk_files - coco_files)
    return missing_on_disk, extra_on_disk


def images_without_annotations(
    df_images: pd.DataFrame, df_annotations: pd.DataFrame
) -> pd.DataFrame:
    """
    Images présentes dans COCO mais sans aucune annotation.
    df_images doit contenir 'id' ; df_annotations doit contenir 'image_id'.
    """
    return df_images[
        ~df_images["id"].isin(pd.unique(df_annotations["image_id"]))
    ].copy()


def annotations_without_image(
    df_annotations: pd.DataFrame, df_images: pd.DataFrame
) -> pd.DataFrame:
    """
    Annotations dont 'image_id' ne correspond à aucune image.
    """
    return df_annotations[
        ~df_annotations["image_id"].isin(pd.unique(df_images["id"]))
    ].copy()


def find_annotation_outliers(df_annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Bboxes aberrantes (w<=0, h<=0, NaN). COCO bbox = [x,y,w,h].
    """
    b = (
        df_annotations["bbox"]
        .apply(lambda bb: {"w": float(bb[2]), "h": float(bb[3])})
        .apply(pd.Series)
    )
    mask = (b["w"] <= 0) | (b["h"] <= 0) | b["w"].isna() | b["h"].isna()
    return df_annotations[mask].copy()
