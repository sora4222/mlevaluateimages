import tensorflow as tf
from pathlib import Path

DATASET_SAVE = "image_dataset.tfrecord"


def ingestion(path_to_images: [str, Path]):
    """
    Ingests the dataset for the pipeline.
    Returns:
        Pipeline with the ingested files.
    """
    pass

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_images_to_tfrecords(path_to_images: [str, Path]):
    """
    Converts images to tfrecords and saves them in the location of the images.
    Args:
        path_to_images:

    Returns:

    """
    path_to_images = Path(path_to_images)
    if not path_to_images.exists() or not path_to_images.is_dir():
        raise NotADirectoryError(f"The directory {path_to_images} does not exist.")
    save_path = path_to_images.joinpath(DATASET_SAVE)
    if save_path.exists():
        save_path.unlink()
    dirs = []
    for path in path_to_images.iterdir():
        if path.is_dir():
            dirs.append(path.name)
    writer: tf.io.TFRecordWriter
    with tf.io.TFRecordWriter(path_to_images.joinpath(DATASET_SAVE)) as writer:
        for label in dirs:
            for img_path in path_to_images.joinpath(label).iterdir():
                try:
                    img_file: tf.Tensor = tf.io.read_file(str(img_path))
                except FileNotFoundError:
                    print(f"File {img_path} could not be found.")
                    continue
                img_tfrecord = tf.train.Example(
                    features=tf.train.Features(
                        feature={'image_raw': _bytes_feature(img_file.numpy()),
                                 'label': _int64_feature(label)}
                    )
                )
                writer.write(img_tfrecord)
