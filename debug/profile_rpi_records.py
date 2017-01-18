from __future__ import print_function
from __future__ import absolute_import

import os
import argparse

import h5py


def _list_files(directory):
    return [
        os.path.join(directory, file_)
        for file_ in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file_))
    ]


def _parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='View data recorded with RPiRoverRecorder env.'
    )
    parser.add_argument(
        'input_directory'
    )
    return parser.parse_args()


def _main():
    args = _parse_command_line_arguments()
    for filepath in _list_files(args.input_directory):
        file_ = h5py.File(filepath, mode='r')
        for value in file_.values():
            print(filepath, value.shape[0])
            break


if __name__ == '__main__':
    _main()
