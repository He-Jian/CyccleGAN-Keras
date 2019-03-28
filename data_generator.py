from PIL import Image
import random
import os
from config import *

np.random.seed(seed)


def image_generator(a_path, b_path, batch_size, shuffle=True):
    image_filenames_A = os.listdir(a_path)
    image_filenames_B = os.listdir(b_path)
    n_batch = len(image_filenames_A) / batch_size
    while True:
        if shuffle:
            random.shuffle(image_filenames_A)
        for i in range(n_batch):
            a_batch = []
            b_batch = []
            for j in range(batch_size):
                index = i * batch_size + j
                a = Image.open(os.path.join(a_path, image_filenames_A[index])).convert('RGB')
                if shuffle:
                    index_b = np.random.randint(0, len(image_filenames_B))
                else:
                    index_b = index % len(image_filenames_B)
                b = Image.open(os.path.join(b_path, image_filenames_B[index_b])).convert('RGB')
                a = a.resize((crop_from, crop_from), Image.BICUBIC)
                b = b.resize((crop_from, crop_from), Image.BICUBIC)
                if random.random() < 0.5:
                    a = a.transpose(Image.FLIP_LEFT_RIGHT)
                    b = b.transpose(Image.FLIP_LEFT_RIGHT)
                a = np.asarray(a) / 127.5 - 1
                b = np.asarray(b) / 127.5 - 1
                w_offset = np.random.randint(0, max(0, crop_from - image_size - 1)) if shuffle else (
                                                                                                    crop_from - image_size) / 2
                h_offset = np.random.randint(0, max(0, crop_from - image_size - 1)) if shuffle else (
                                                                                                    crop_from - image_size) / 2
                a = a[h_offset:h_offset + image_size, w_offset:w_offset + image_size, :]
                b = b[h_offset:h_offset + image_size, w_offset:w_offset + image_size, :]
                a_batch.append(a)
                b_batch.append(b)
            yield np.array(a_batch), np.array(b_batch)
            # yield (np.array(a_batch) - imagenet_mean) / imagenet_std, (np.array(b_batch) - imagenet_mean) / imagenet_std
