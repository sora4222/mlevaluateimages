from pathlib import Path
import tensorflow as tf
from tfx.components import ImportExampleGen

DATASET_SAVE = "image_dataset.tfrecord"


def ingestion(records_path: [str, Path], path_to_images: [str, Path]):
    """
    Ingests the dataset for the pipeline.
    Returns:
        Pipeline with the ingested files.
    """
    path_to_images = Path(path_to_images)
    records_path = Path(records_path)
    # When the path is a directory, convert the images to tfrecords
    if not records_path.joinpath(DATASET_SAVE).exists():
        convert_images_to_tfrecords(path_to_images, records_path)
    example_gen = ImportExampleGen(input_base=str(records_path))
    return example_gen



def convert_images_to_tfrecords(path_to_images: [str, Path], records_path: Path):
    """
    Converts images to tfrecords and saves them in the location of the images by default to a file
    called `image_dataset.tfrecord`.

    Args:
        records_path: Path to save the tfrecord file
        path_to_images: Parent directory of the directories with images

    Notes:
        This is used instead of using `FileBasedExampleGen` from `tfx.components`

    Returns:
        The path of the tfrecord file
    """
    path_to_images = Path(path_to_images)
    if not path_to_images.exists() or not path_to_images.is_dir():
        raise NotADirectoryError(f"The directory {path_to_images} does not exist.")
    save_path = records_path.joinpath(DATASET_SAVE)
    if save_path.exists():
        save_path.unlink()
    dirs = []
    for path in path_to_images.iterdir():
        if path.is_dir():
            dirs.append(path.name)
    writer: tf.io.TFRecordWriter
    with tf.io.TFRecordWriter(str(save_path)) as writer:
        for index, label in enumerate(sorted(dirs)):
            for img_path in path_to_images.joinpath(label).iterdir():
                try:
                    img_file: tf.Tensor = tf.io.read_file(str(img_path))
                except FileNotFoundError:
                    print(f"File {img_path} could not be found.")
                    continue
                img_tfrecord = tf.train.Example(
                    features=tf.train.Features(
                        feature={'image_raw': _bytes_feature(img_file.numpy()),
                                 'label': _int64_feature(index)}
                    )
                )
                writer.write(img_tfrecord.SerializeToString())

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
