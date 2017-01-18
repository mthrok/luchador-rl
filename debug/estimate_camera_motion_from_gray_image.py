from __future__ import print_function

import time
import argparse

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt


def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, img1.dtype)

    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color

    for match in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            shape = (0, 256, 3) if img1.ndim == 3 else (0, 256)
            c = np.random.randint(*shape)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img


def _convert(array, dtype='float32', shape=(-1, 1, 2)):
    return np.asarray(array, dtype=dtype).reshape(*shape)


def _process(img1, img2, min_good_matches=10):
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print('#KeyPoints  : ', len(kp1), len(kp2))

    flann_index_kdtree = 0
    flann = cv2.FlannBasedMatcher(
        indexParams={'algorithm': flann_index_kdtree, 'trees': 5},
        searchParams={'checks': 50}
    )

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = [
        match for match, norm in matches
        if match.distance < 0.7 * norm.distance
    ]

    if len(good_matches) <= min_good_matches:
        print(
            'Not enough matches are found - {} / {}'
            .format(len(good_matches), min_good_matches)
        )
    else:
        src_pts = _convert([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = _convert([kp2[m.trainIdx].pt for m in good_matches])

        F, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)
        # TODO Convert Fundamental Matrix to Essential Matrix then retrieve
        # parallel motion.

    return draw_matches(
        img1, kp1, img2, kp2, good_matches, color=(0, 255, 0)
    )


def _parse_command_line_args():
    parser = argparse.ArgumentParser(
        description=(
            'Test camera motion estimation using SHIFT and '
            'RANSAC over gray images from RPiRecorder data'
        )
    )
    parser.add_argument('input_file', help='Input HDF5 file.')
    return parser.parse_args()


def _normalize(array):
    array = array.astype('float32')
    min_ = array.min()
    return (array - min_) / (array.max() - min_)


def _load_image(filepath, image_type='gray_image'):
    file_ = h5py.File(filepath, 'r')
    data = 255 * _normalize(np.copy(file_[image_type]))
    file_.close()
    return data.astype('uint8')


def _main():
    args = _parse_command_line_args()
    data = _load_image(args.input_file)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img_ = None

    for i in range(data.shape[0] - 1):
        img = _process(data[i], data[i+1])

        if img_ is None:
            ax.imshow(img, 'gray_r')
            plt.show(block=False)
        else:
            img_.set_data(img)

        fig.canvas.draw()
        time.sleep(0.3)


if __name__ == '__main__':
    _main()
