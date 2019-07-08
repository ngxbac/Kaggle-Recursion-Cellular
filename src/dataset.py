import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset

NUM_CLASSES = 1108

DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}



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
        sites number
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


def image_stats(pixel_stat,
               experiment,
               plate,
               address,
               site,
               channel):
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
        sites number
    channel : int
        channel number
    base_path : str
        the base path of the raw images
    Returns
    -------
    str the path of image
    """

    channel_stat = pixel_stat[(pixel_stat.experiment == experiment)
               & (pixel_stat.plate == plate)
               & (pixel_stat.well == address)
               & (pixel_stat.site == site)
               & (pixel_stat.channel == channel)]

    return channel_stat["mean"].values[0], channel_stat["std"].values[0]

# def load_image(image_path):
#     with tf.io.gfile.GFile(image_path, 'rb') as f:
#         return imread(f, format='png')


def load_images_as_tensor(image_paths, dtype=np.uint8):
    n_channels = len(image_paths)

    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path)

    return data


def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the sites as RGB channels
    """
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) / \
            ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
            rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


class RecursionCellularSite(Dataset):

    def __init__(self,
                 csv_file,
                 root,
                 transform,
                 sites=[1],
                 mode='train',
                 channels=[1, 2, 3, 4, 5, 6],
                 ):
        print("Channels ", channels)
        print("sites ", sites)
        df = pd.read_csv(csv_file, nrows=None)
        self.pixel_stat = pd.read_csv(os.path.join(root, "pixel_stats.csv"))
        self.stat_dict = {}
        for experiment, plate, well, site, channel, mean, std in zip(self.pixel_stat.experiment,
                                                                   self.pixel_stat.plate,
                                                                   self.pixel_stat.well,
                                                                   self.pixel_stat.site,
                                                                   self.pixel_stat.channel,
                                                                   self.pixel_stat["mean"],
                                                                   self.pixel_stat["std"]):
            if not experiment in self.stat_dict:
                self.stat_dict[experiment] = {}

            if not plate in self.stat_dict[experiment]:
                self.stat_dict[experiment][plate] = {}

            if not well in self.stat_dict[experiment][plate]:
                self.stat_dict[experiment][plate][well] = {}

            if not site in self.stat_dict[experiment][plate][well]:
                self.stat_dict[experiment][plate][well][site] = {}

            if not channel in self.stat_dict[experiment][plate][well][site]:
                self.stat_dict[experiment][plate][well][site][channel] = {}

            self.stat_dict[experiment][plate][well][site][channel]["mean"] = mean / 255
            self.stat_dict[experiment][plate][well][site][channel]["std"] = std / 255


        self.transform = transform
        self.mode = mode
        self.channels = channels
        self.sites = sites

        self.experiments = df['experiment'].values
        self.plates = df['plate'].values
        self.wells = df['well'].values

        if mode != 'test':
            self.labels = df['sirna'].values
        else:
            self.labels = [0] * len(self.experiments)

        self.root = root

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):

        experiment = self.experiments[idx]
        plate = self.plates[idx]
        well = self.wells[idx]

        channel_paths = []

        if self.mode == 'train':
            if np.random.rand() < 0.5:
                sites = [1]
            else:
                sites = [2]
        else:
            sites = self.sites

        for site in sites:
            for channel in self.channels:
                path = image_path(
                    dataset=self.mode,
                    experiment=experiment,
                    plate=plate,
                    address=well,
                    channel=channel,
                    site=site,
                    base_path=self.root,
                )
                channel_paths.append(path)

        std_arr = []
        mean_arr = []

        for site in sites:
            for channel in self.channels:
                mean = self.stat_dict[experiment][plate][well][site][channel]["mean"]
                std = self.stat_dict[experiment][plate][well][site][channel]["std"]
                std_arr.append(std)
                mean_arr.append(mean)

        image = load_images_as_tensor(channel_paths, dtype=np.float32)
        # image = convert_tensor_to_rgb(image)
        # image = image / 255
        if self.transform:
            image = self.transform(image=image)['image']

        image = normalize(image, std=std_arr, mean=mean_arr, max_pixel_value=255)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.mode == 'train':
            label = self.labels[idx]
        else:
            label = -1

        return {
            "images": image,
            "targets": label
        }


from sklearn.preprocessing import MultiLabelBinarizer
MEAN = [0.08069, 0.05258, 0.05487, 0.08282]
STD = [0.13704, 0.10145, 0.15313, 0.13814]
classes = [
    '0', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '10', '11', '12', '13',
    '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27'
]


def load_image_hpa(path):
    image_red_ch = cv2.imread(path + '_red.png', 0)
    image_green_ch = cv2.imread(path + '_green.png', 0)
    image_blue_ch = cv2.imread(path + '_blue.png', 0)
    image_yellow_ch = cv2.imread(path + '_yellow.png', 0)

    # RGBY image
    image = cv2.merge((
        image_red_ch,
        image_green_ch,
        image_blue_ch,
        image_yellow_ch,
    ))

    return image / 255


class ProteinDataset(Dataset):
    def __init__(self, root, root_hpa, df, transform=None, mode="train"):
        df = pd.read_csv(df)
        self.root = root
        self.root_hpa = root_hpa
        self.ids = df["Id"].values
        if "is_external" in df.columns.values:
            self.is_external = df["is_external"].values
        else:
            self.is_external = [False] * len(df)

        self.mlb = MultiLabelBinarizer()
        if mode == "train":
            self.mlb.classes_ = classes
            self.labels = self.mlb.transform(df['Target'].str.split()).astype(np.float32)
            print(self.mlb.classes_)

        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        id = self.ids[idx]
        is_external = self.is_external[idx]
        if is_external:
            path = os.path.join(self.root_hpa, id)
        else:
            path = os.path.join(self.root, id)
        image = load_image_hpa(path)

        if self.transform:
            image = self.transform(image=image)["image"]

            # Normalize per channel
            mean = np.mean(image.reshape(-1, 4), axis=0)
            std = np.std(image.reshape(-1, 4), axis=0)
            image -= mean
            image /= (std + 0.0000001)

            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.mode == "train":
            label = self.labels[idx]
        else:
            label = np.asarray([0] * 28)

        return {
            "images": image,
            "targets": label,
        }
