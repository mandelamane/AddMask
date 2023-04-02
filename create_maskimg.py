import os
import tarfile

import tensorflow as tf

from utils import download_data
from utils.configuration import Configuration
from utils.data_generator import DataGenerator

gpu_physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(gpu_physical_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_physical_devices[0], True)

# check HW availability
print(
    "Num GPUs Available: ",
    len(tf.config.experimental.list_physical_devices("GPU")),
)
print(
    "Num CPUs Available: ",
    len(tf.config.experimental.list_physical_devices("CPU")),
)

configuration = Configuration()
dataset_path = configuration.get("input_images_path")
if os.path.isdir(dataset_path):
    print("Dataset already downloaded")
else:
    print("Downloading dataset")
    dataset_archive_path = os.path.join("data", "lfw-deepfunneled.tgz")
    download_data(
        configuration.get("dataset_archive_download_url"), dataset_archive_path
    )
    print("Extracting dataset")
    tar = tarfile.open(dataset_archive_path, "r:gz")
    tar.extractall("data")
    tar.close()
    print("Done")

dg = DataGenerator(configuration)

train_folder = configuration.get("train_data_path")
test_folder = configuration.get("test_data_path")

if os.path.exists(train_folder) and os.path.exists(test_folder):
    print("Testing and training data already generated")
else:
    dg.generate_images()
