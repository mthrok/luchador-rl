from __future__ import print_function
from __future__ import absolute_import

import time
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='View data recorded with RPiRoverRecorder env.'
    )
    parser.add_argument(
        'input_file'
    )
    parser.add_argument(
        '--plot', action='store_true',
    )
    return parser.parse_args()


def _normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def _play_depth_image(
        gray_images, x_image, y_image, z_image):
    fig = plt.figure()
    axes = [fig.add_subplot(2, 2, i+1) for i in range(4)]
    imgs = [
        axes[0].imshow(gray_images[0, ...], cmap='gray_r'),
        axes[1].imshow(
            x_image[0, ...], cmap='jet_r',
            vmin=x_image.min(), vmax=x_image.max()),
        axes[2].imshow(
            y_image[0, ...], cmap='jet_r',
            vmin=y_image.min(), vmax=y_image.max()),
        axes[3].imshow(
            z_image[0, ...], cmap='jet_r',
            vmin=z_image.min(), vmax=z_image.max()),
    ]
    plt.show(block=False)

    while True:
        for i in range(gray_images.shape[0]):
            imgs[0].set_data(gray_images[i, ...])
            imgs[1].set_data(x_image[i, ...])
            imgs[2].set_data(y_image[i, ...])
            imgs[3].set_data(z_image[i, ...])
            fig.canvas.draw()
            time.sleep(0.1)


def _load_data(filepath):
    file_ = h5py.File(filepath, mode='r')
    for key, value in file_.items():
        print(key, value.dtype, value.shape)
    ret = {
        'gray_images': np.copy(file_['gray_image']),
        'x_image': np.copy(file_['depth_image_x']),
        'y_image': np.copy(file_['depth_image_y']),
        'z_image': np.copy(file_['depth_image_z']),
    }
    file_.close()
    return ret


def _main():
    args = _parse_command_line_arguments()
    data = _load_data(args.input_file)
    try:
        if args.plot:
            _play_depth_image(**data)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    _main()
