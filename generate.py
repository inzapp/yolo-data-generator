import os
from glob import glob
from random import shuffle

import cv2
import numpy as np

generate_count = 2000


def multiple_width_height(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(x1 * width)
    x2 = int(x2 * width)
    y1 = int(y1 * height)
    y2 = int(y2 * height)
    return [x1, y1, x2, y2]


def to_x1_y1_x2_y2(box):
    cx, cy, w, h = box
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    box = [x1, y1, x2, y2]
    box = list(np.clip(np.asarray(box), 0.0, 1.0))
    return box


def adjust(img, adjust_type):
    weight = np.random.uniform(0.75, 1.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    if adjust_type == 'hue':
        h = np.asarray(h).astype('float32') * weight
        h = np.clip(h, 0.0, 255.0).astype('uint8')
    elif adjust_type == 'saturation':
        s = np.asarray(s).astype('float32') * weight
        s = np.clip(s, 0.0, 255.0).astype('uint8')
    elif adjust_type == 'brightness':
        v = np.asarray(v).astype('float32') * weight
        v = np.clip(v, 0.0, 255.0).astype('uint8')
    elif adjust_type == 'contrast':
        weight = np.random.uniform(0.0, 0.25)
        criteria = np.random.uniform(84.0, 170.0)
        v = np.asarray(v).astype('float32')
        v += (criteria - v) * weight
        v = np.clip(v, 0.0, 255.0).astype('uint8')
    elif adjust_type == 'noise':
        range_min = np.random.uniform(0.0, 25.0)
        range_max = np.random.uniform(0.0, 25.0)
        v = np.asarray(v).astype('float32')
        v += np.random.uniform(-range_min, range_max, size=v.shape)
        v = np.clip(v, 0.0, 255.0).astype('uint8')

    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def get_save_path(img_path, inc):
    save_dir_name = 'generated'
    if not (os.path.exists(save_dir_name) and os.path.isdir(save_dir_name)):
        os.makedirs(save_dir_name, exist_ok=True)
    img_path = img_path.replace('\\', '/')
    raw_file_name = img_path.split('/')[-1][:-4]
    new_file_name = f'generated_{raw_file_name}_{inc}'
    img_path = img_path.replace(raw_file_name, new_file_name)
    label_path = f'{img_path[:-4]}.txt'
    img_path = f'{save_dir_name}/{img_path}'
    label_path = f'{save_dir_name}/{label_path}'
    return img_path, label_path


def generate(image_paths):
    global generate_count
    if len(image_paths) == 0:
        return

    inc = 0
    shuffle(image_paths)
    while True:
        for path in image_paths:
            label_path = f'{path[:-4]}.txt'
            if not (os.path.exists(label_path) and os.path.isfile(label_path)):
                continue

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            with open(label_path, 'rt') as f:
                label_lines = f.readlines()

            adjust_opts = ['hue', 'saturation', 'brightness', 'contrast', 'noise']
            shuffle(adjust_opts)
            for i in range(len(adjust_opts)):
                img = adjust(img, adjust_opts[i])

            new_img_path, new_label_path = get_save_path(path, inc)
            cv2.imwrite(new_img_path, img)
            with open(new_label_path, 'wt') as f:
                f.writelines(label_lines)

            inc += 1
            print(f'[{inc:5d}/{generate_count:5d}] saved ===> {new_img_path}')
            if inc >= generate_count:
                return


def get_num_classes():
    with open('classes.txt', 'rt') as f:
        classes = f.readlines()
    return len(classes)


def get_class_counts(image_paths, num_classes):
    class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
    for path in image_paths:
        label_path = f'{path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            continue

        with open(label_path, 'rt') as f:
            lines = f.readlines()

        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            class_index = int(class_index)
            class_counts[class_index] += 1
    return list(class_counts)


def print_class_counts():
    image_paths = glob('*.jpg')
    num_classes = get_num_classes()
    class_counts = get_class_counts(image_paths, num_classes)
    for class_index in range(num_classes):
        print(f'class {class_index:3d} : {class_counts[class_index]:6d}')


if __name__ == '__main__':
    print_class_counts()
    generate(glob('*.jpg'))
