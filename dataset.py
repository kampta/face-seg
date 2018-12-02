import random
import re
from glob import glob

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def _mask_to_img(mask_file):
    img_file = re.sub('masks', 'images', mask_file)
    img_file = re.sub('\.ppm$', '.jpg', img_file)
    return img_file


class MaskDataset(Dataset):
    def __init__(self, path='./data', transform=None):

        self.mask_files = sorted(glob('{}/masks/*.ppm'.format(path)))
        self.img_files = [_mask_to_img(f) for f in self.mask_files]
        self.transform = transform

    def __getitem__(self, idx):
        # print(self.img_files[idx])
        # print(self.mask_files[idx])

        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx])
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = mask[:, :, 0: 2]

        seed = random.randint(0, 2 ** 32)

        if self.transform is not None:
            # Apply transform to img
            random.seed(seed)
            img = Image.fromarray(img)
            img = self.transform(img)

            # Apply same transform to mask
            random.seed(seed)
            mask = Image.fromarray(mask)
            mask = self.transform(mask)

        mask = np.array(mask)
        # print(mask.shape)
        labels = np.zeros_like(mask[0, :, :])
        labels[np.where(mask[1, :, :] > 0)] = 1  # hair
        labels[np.where(mask[2, :, :] > 0)] = 2  # face
        # labels = np.expand_dims(labels, axis=0)

        return img, np.int64(labels)

    def __len__(self):
        return len(self.img_files)


def gen_dataloaders(indir, val_split=0.05, shuffle=True,
                    batch_size=4, seed=42, img_size=224, cuda=True):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomResizedCrop(img_size, scale=(0.1, 1.0)),
            transforms.RandomAffine(10.),
            transforms.RandomRotation(13.),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    }

    mask_train_dataset = MaskDataset(path=indir,
                                     transform=data_transforms['train'])
    mask_valid_dataset = MaskDataset(path=indir,
                                     transform=data_transforms['valid'])

    # Creating data indices for training and validation splits:
    dataset_size = len(mask_train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(mask_train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(mask_valid_dataset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               **kwargs)
    return train_loader, valid_loader


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='LFW dataset')

    # Arguments
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')

    args = parser.parse_args()

    train_loader, valid_loader = gen_dataloaders(args.data_folder,
                                                 batch_size=args.batch_size)

    images, masks = next(iter(train_loader))
    # print(images.size())
    # print(masks.size())
    fig = plt.figure()

    images = images.permute(0, 2, 3, 1).numpy()
    # masks = masks.permute(0, 2, 3, 1).numpy()

    for i, (image, mask) in enumerate(zip(images, masks)):
        print(i, image.shape, mask.shape)
        # print(np.unique(mask))
        ax = plt.subplot(args.batch_size, 2, 2 * i + 1)
        # ax.set_title('image')
        ax.axis('off')
        ax.imshow(image.squeeze())

        ax = plt.subplot(args.batch_size, 2, 2 * i + 2)
        # ax.set_title('mask')
        ax.axis('off')
        ax.imshow(mask.squeeze())

    plt.show()
