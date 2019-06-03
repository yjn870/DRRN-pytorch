import argparse
import glob
import h5py
import numpy as np
from utils import load_image, modcrop, generate_lr, image_to_array, rgb_to_y, normalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = load_image(image_path)
        hr = modcrop(hr, args.scale)
        lr = generate_lr(hr, args.scale)

        hr = image_to_array(hr)
        lr = image_to_array(lr)

        hr = np.expand_dims(normalize(rgb_to_y(hr.astype(np.float32), 'chw')), 0)
        lr = np.expand_dims(normalize(rgb_to_y(lr.astype(np.float32), 'chw')), 0)

        hr_group.create_dataset(str(i), data=hr)
        lr_group.create_dataset(str(i), data=lr)

    h5_file.close()
