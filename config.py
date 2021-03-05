from dataclasses import dataclass
from typing import Tuple


# TODO: adjust values
@dataclass
class TrainerConfig:
    # the path to the folder containing dataset of content images.
    # this folder should be structured as follows:
    # - train
    #   - image_1.jpg
    #   - ...
    # - validation
    #   - image_1.jpg
    #   - ...
    content_dataset_path: str = r'D:\Downloads\ms-coco-2014'

    # the path to the folder containing dataset of style images.
    # this folder should be structured as follows:
    # - train
    #   - image_1.jpg
    #   - ...
    # - validation
    #   - image_1.jpg
    #   - ...
    style_dataset_path: str = r'D:\Downloads\wiki-art'

    # the root directory in which model snapshots
    # and TensorBoard logs will be saved.
    # root_logdir: str = 'models'
    root_logdir: str = r'd:\models'

    # a path to a snapshot of the model's weights.
    # to be used when resuming a previous training job.
    weights_snapshot_path: str = ''

    # the weight of the content term in the total loss.
    # empirically good range: 1 - 100
    lambda_content: float = 1.0

    # the weight of the style term in the total loss.
    # empirically good range: 10 - 100_000
    lambda_style: float = 0.01

    # the weight of the generated image's total variation
    # in the total loss. empirically good range: 0 - 1_000.
    lambda_tv: float = 0.0

    # the size of each step of the optimization process.
    learning_rate: float = 1e-4

    # the scaling factor to apply to the learning rate
    # on the end of each epoch.
    lr_gamma: float = 0.85

    # number of training epochs to perform.
    epochs: int = 8

    # the dimension of the model's input images.
    input_images_dim: int = 256

    # the interval (number of training iterations) after which intermediate
    # results of the stylized images will be visualized in TensorBoard.
    visualization_interval: int = 50

    # the interval (number of training iterations) after which an
    # intermediate snapshot of the model will be saved to the disk.
    snapshot_interval: int = 5000

    # the mini batch size to use for each training iteration.
    batch_size: int = 4

    # the number of workers to use for loading images
    # form the dataset in the background
    num_data_loader_workers: int = 5

    def update(self, **kwargs) -> 'TrainerConfig':
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise KeyError(f'Unknown configuration value: "{key}"')
        return self
