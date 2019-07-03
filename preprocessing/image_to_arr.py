import pandas as pd
import numpy as np
import cv2
import os

import click
from tqdm import *


def load_image(path):
    image = cv2.imread(path, 0)
    return image


def image_path(dataset,
               experiment,
               plate,
               address,
               site,
               channel,
               base_path):
    """
    Returns the path of a channel image.
    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channel : int
        channel number
    base_path : str
        the base path of the raw images
    Returns
    -------
    str the path of image
    """
    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))


def load_images_as_tensor(image_paths, dtype=np.uint8):
    n_channels = len(image_paths)

    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path)

    return data


@click.group()
def cli():
    print("Convert images to array")


@cli.command()
@click.option('--csv', type=str)
@click.option('--base_path', type=str)
@click.option('--output', type=str)
@click.option('--dataset', type=str)
def image_to_arr(
        csv,
        base_path,
        output,
        dataset,
    ):
    channels = [1, 2, 3, 4, 5, 6]
    df = pd.read_csv(csv)
    experiments = df['experiment'].values
    plates = df['plate'].values
    wells = df['well'].values

    import pdb
    pdb.set_trace()

    for experiment, plate, well in tqdm(zip(experiments, plates, wells), total=len(experiments)):
        for site in [1, 2]:
            channel_paths = [
                image_path(
                    dataset=dataset,
                    experiment=experiment,
                    plate=plate,
                    address=well,
                    channel=channel,
                    site=site,
                    base_path=base_path,
                ) for channel in channels
            ]
            image = load_images_as_tensor(channel_paths, dtype=np.float32)
            os.makedirs(
                os.path.join(output, dataset, experiment, "Plate{}".format(plate)),
                exist_ok=True
            )
            np.save(
                os.path.join(output, dataset, experiment, "Plate{}".format(plate), "{}_s{}.npy".format(well, site)),
                image
            )


if __name__ == '__main__':
    cli()
