import sys
from os.path import isfile
import numpy as np
import cv2


def load_raw_image(filename: str) -> bytes:

    with open(filename, 'rb') as fin:
        raw_image = fin.read()

    return raw_image


def get_raw_image_info(raw_image: bytes) -> tuple:

    raw_image_size = len(raw_image)
    raw_image_width = raw_image_height = int((raw_image_size / 3) ** 0.5)

    if raw_image_width * raw_image_height * 3 != raw_image_size:
        raw_image_width = raw_image_size
        raw_image_height = 1

    return (raw_image_height, raw_image_width)


def resolve_channels_order(raw_image_height: int, raw_image_width: int, raw_image: bytes) -> np.array:

    image = np.zeros(
        shape=[raw_image_height, raw_image_width, 3], dtype=np.uint8)

    # (RGB)^n
    for i in range(raw_image_height):
        for j in range(raw_image_width):
            for k in range(3):
                try:
                    image[i][j][k] = raw_image[i*raw_image_width*3 + j*3 + k]

                except IndexError as e:
                    print(
                        'IndexError at ({}, {}, {}) during resolving raw image.'.format(j, i, k))
                    print(e)
                    exit(1)

    return image


def resolve_mono_channel_order(raw_image_height: int, raw_image_width: int, raw_image: bytes) -> np.array:

    image = np.zeros(
        shape=[raw_image_height, raw_image_width, 3], dtype=np.uint8)

    # (R)^n  (G)^n  (B)^n
    for i in range(3):
        for j in range(raw_image_height):
            for k in range(raw_image_width):
                try:
                    image[j][k][i] = raw_image[i*raw_image_width *
                                               raw_image_height + j*raw_image_width + k]

                except IndexError as e:
                    print(
                        'IndexError at ({}, {}, {}) during resolving raw image.'.format(j, k, i))
                    print(e)
                    exit(1)

    return image


def write_image(image: np.array, filename: str) -> bool:

    return cv2.imwrite(filename, image)


if __name__ == '__main__' and len(sys.argv) >= 2:
    filename = sys.argv[1]
    output_filename = filename

    if isfile(filename):
        raw_image = load_raw_image(filename)
        print('Image Loaded.')

        (raw_image_height, raw_image_width) = get_raw_image_info(raw_image)
        print('Image Size: (w = {}, h = {})'.format(
            raw_image_width, raw_image_height))

        if output_filename.endswith('.raw'):
            output_filename = output_filename[0:-4] + '.jpg'

        write_image(resolve_channels_order(raw_image_height,
                    raw_image_width, raw_image), output_filename)
        print('Done resolving.')

    else:
        print('File NOT found: {}'.format(filename))

else:
    print('Usage: pass filename to be resolved as first argument.\n')
