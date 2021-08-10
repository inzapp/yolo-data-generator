import os
from glob import glob
from random import shuffle

import cv2
import numpy as np

target_class_count = 1000


def remove_other_class(img, label_lines, target_class_index):
    raw_height, raw_width = img.shape[0], img.shape[1]
    global_mean_bgr = np.mean(np.mean(img, axis=1), axis=0)
    new_label_lines = []
    for line in label_lines:
        raw_line = line
        class_index, cx, cy, w, h = list(map(float, line.replace('\n', '').split()))
        if class_index == target_class_index:
            new_label_lines.append(raw_line)
            continue

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1 = int(x1 * raw_width)
        y1 = int(y1 * raw_height)
        x2 = int(x2 * raw_width)
        y2 = int(y2 * raw_height)

        for y in range(y1, y2):
            for x in range(x1, x2):
                img[y][x] = global_mean_bgr
    return img, new_label_lines, len(new_label_lines)


def adjust(img, range_min, range_max, adjust_type):
    weight = np.random.uniform(range_min, range_max, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    if adjust_type == 'brightness':
        v = np.asarray(v).astype('float32') * weight
        v = np.clip(v, 0.0, 255.0).astype('uint8')
    elif adjust_type == 'saturation':
        s = np.asarray(s).astype('float32') * weight
        s = np.clip(s, 0.0, 255.0).astype('uint8')
    elif adjust_type == 'hue':
        h = np.asarray(h).astype('float32') * weight
        h = np.clip(h, 0.0, 255.0).astype('uint8')

    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def adjust_brightness(img, range_start, range_end):
    weight = np.random.uniform(range_start, range_end, 1)
    img = np.asarray(img).astype('float32') * weight
    img = np.clip(img, 0.0, 255.0).astype('uint8')
    return img


def adjust_saturation(img, range_start, range_end):
    weight = np.random.uniform(range_start, range_end, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    s = np.asarray(s).astype('float32') * weight
    s = np.clip(s, 0.0, 255.0).astype('uint8')
    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def adjust_hue(img, range_start, range_end):
    weight = np.random.uniform(range_start, range_end, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    h = np.asarray(h).astype('float32') * weight
    h = np.clip(h, 0.0, 255.0).astype('uint8')
    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def get_save_path(img_path, class_index, inc):
    img_path = img_path.replace('\\', '/')
    raw_file_name = img_path.split('/')[-1][:-4]
    new_file_name = f'generated_{class_index}_{inc}_{raw_file_name}'
    img_path = img_path.replace(raw_file_name, new_file_name)
    label_path = f'{img_path[:-4]}.txt'
    return img_path, label_path


def generate(class_image_paths, class_count, class_index):
    global target_class_count
    if class_count >= target_class_count:
        return

    inc = 0
    shuffle(class_image_paths)
    while True:
        for path in class_image_paths:
            label_path = f'{path[:-4]}.txt'
            if not (os.path.exists(label_path) and os.path.isfile(label_path)):
                continue

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            with open(label_path, 'rt') as f:
                label_lines = f.readlines()

            img = adjust(img, 0.75, 1.25, 'hue')
            img = adjust(img, 0.75, 1.25, 'saturation')
            img = adjust(img, 0.75, 1.25, 'brightness')
            img, label_lines, cur_image_class_count = remove_other_class(img, label_lines, class_index)

            new_img_path, new_label_path = get_save_path(path, class_index, inc)
            cv2.imwrite(new_img_path, img)
            with open(new_label_path, 'wt') as f:
                f.writelines(label_lines)

            print(f'saved ===> {new_img_path}')

            inc += 1
            class_count += cur_image_class_count
            if class_count >= target_class_count:
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


def get_class_image_paths(image_paths, num_classes):
    class_image_paths = [[] for _ in range(num_classes)]
    for path in image_paths:
        label_path = f'{path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            continue

        with open(label_path, 'rt') as f:
            lines = f.readlines()

        # ignore duplicate
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            class_index = int(class_index)
            class_image_paths[class_index].append(path)

    # remove duplicate
    for class_index in range(num_classes):
        class_image_paths[class_index] = list(set(class_image_paths[class_index]))
    return class_image_paths


def print_class_counts():
    image_paths = glob('*.jpg')
    num_classes = get_num_classes()
    class_counts = get_class_counts(image_paths, num_classes)
    for class_index in range(num_classes):
        print(f'class {class_index:3d} : {class_counts[class_index]:6d}')


def class_balanced_generate():
    image_paths = glob('*.jpg')
    num_classes = get_num_classes()

    class_counts = get_class_counts(image_paths, num_classes)
    class_image_paths = get_class_image_paths(image_paths, num_classes)

    for class_index in range(num_classes):
        generate(class_image_paths[class_index], class_counts[class_index], class_index)


if __name__ == '__main__':
    print_class_counts()
    class_balanced_generate()
