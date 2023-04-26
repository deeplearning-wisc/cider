import os
import shutil
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser(description='Create ImageNet-100 subset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--K', default=100, type=int, help='num of classes to be subsampled')
parser.add_argument('--src-dir', default='/nobackup/imagenet', type=str,
                    help='path to ImageNet-1k')
                    # '/path/to/ImageNet-1k'
parser.add_argument('--dst-dir', default='datasets/imagenet-100', type=str,
                    help='root dir of in_dataset')
args = parser.parse_args()
os.makedirs(args.dst_dir, exist_ok=True)

#subsample K classes from ImageNet-1k
class_names = random.sample(os.listdir(os.path.join(args.src_dir, 'train')), args.K)

for split in ['train', 'val']:
    for cls in tqdm(class_names):
        shutil.copytree(os.path.join(args.src_dir, split, cls), os.path.join(args.dst_dir, split, cls), dirs_exist_ok=True)
    print(f'### Created imagenet-{args.K} {split} ###')