from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist


def best_fit_transform(cloud1, cloud2):
    """Calculates transform between 3D points point cloud 1 to 2

    Parameters
    ----------
      cloud1: Nx3 numpy array of corresponding 3D points
      cloud2: Nx3 numpy array of corresponding 3D points

    Returns
    -------
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    """
    assert len(cloud1) == len(cloud2)

    # translate points to their centroids
    centroid_1 = np.mean(cloud1, axis=0)
    centroid_2 = np.mean(cloud2, axis=0)
    _cloud1 = cloud1 - centroid_1
    _cloud2 = cloud2 - centroid_2

    # rotation matrix
    H = np.dot(_cloud1.T, _cloud2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_2.T - np.dot(R, centroid_1.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """Find the nearest (Euclidean) neighbor in dst for each point in src
    Parameters
    ----------
        src: Nx3 array of points
        dst: Nx3 array of points

    Returns
    -------
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """
    all_dists = cdist(src, dst, 'euclidean')
    indices = all_dists.argmin(axis=1)
    distances = all_dists[np.arange(all_dists.shape[0]), indices]
    return distances, indices


def icp(cloud1, cloud2, init_pose=None, max_iterations=20, tolerance=1e-5):
    """The Iterative Closest Point method
    Parameters
    ----------
        cloud1: Nx3 numpy array of source 3D points
        cloud2: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Returns
    -------
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    """
    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4, cloud1.shape[0]))
    dst = np.ones((4, cloud2.shape[0]))
    src[0:3, :] = np.copy(cloud1.T)
    dst[0:3, :] = np.copy(cloud2.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for ite in range(max_iterations):
        # Find the nearest neighbours
        distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

        # Compute the transformation
        T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = distances.mean()
        res = abs(prev_error-mean_error)
        if res < tolerance:
            break
        prev_error = mean_error

    print('Finished at iteration', ite)
    print('Residual:', res)
    print('Mean error:', mean_error)

    # calculate final transformation
    T, _, _ = best_fit_transform(cloud1, src[0:3, :].T)
    return T, distances


def _parse_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Test camera motion estimation from depth images'
    )
    parser.add_argument(
        'input_file', help='Input HDF5 file created with RPiRoverRecorder'
    )
    return parser.parse_args()


def _load_data(filepath):
    import h5py

    file_ = h5py.File(filepath, 'r')
    x_image = np.copy(file_['depth_image_x'])
    y_image = np.copy(file_['depth_image_y'])
    z_image = np.copy(file_['depth_image_z'])
    file_.close()
    return {
        'x': x_image,
        'y': y_image,
        'z': z_image,
    }


def _save_ply(points, color, filename):
    from plyfile import PlyData, PlyElement

    el_ = []
    for pts in points:
        el_.append(tuple(pts.tolist() + color))
    elem = np.array(el_, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    el = PlyElement.describe(elem, 'vertex')
    PlyData([el], text=True).write(filename)


def _main():
    args = _parse_command_line_args()

    data = _load_data(args.input_file)
    delta = 3
    n_data = len(data['x'])
    for i in range(n_data-delta):
        points1 = np.concatenate((
            data['x'][i, ...].reshape(-1, 1),
            data['y'][i, ...].reshape(-1, 1),
            data['z'][i, ...].reshape(-1, 1),
        ), axis=1)
        points2 = np.concatenate((
            data['x'][i+delta, ...].reshape(-1, 1),
            data['y'][i+delta, ...].reshape(-1, 1),
            data['z'][i+delta, ...].reshape(-1, 1),
        ), axis=1)
        points1 = points1[points1[:, 2] > 0, :]
        points2 = points2[points2[:, 2] > 0, :]

        init_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.08],
             [0., 0., 0., 1.]],
        )
        print(
            'Matching with {} points and {} points'
            .format(len(points1), len(points2))
        )
        T, distances = icp(points2, points1, init_pose=init_pose)
        print('Translation Matrix: \n', T)
        print()

        p2 = np.ones((4, points2.shape[0]))
        p2[0:3, :] = np.copy(points2.T)
        points3 = np.dot(T, p2).T[:, :3]

        _save_ply(
            points1, [255, 0, 0],
            'frame_{:03d}_points1.ply'.format(i)
        )
        _save_ply(
            points2, [0, 255, 0],
            'frame_{:03d}_points2_1.ply'.format(i)
        )
        _save_ply(
            points3, [0, 0, 255],
            'frame_{:03d}_points2_2.ply'.format(i)
        )


if __name__ == '__main__':
    _main()
