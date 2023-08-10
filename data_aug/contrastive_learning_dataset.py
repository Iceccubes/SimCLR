from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets, transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
# Added for SAS evaluation
from PIL import Image
from sas.subset_dataset import SASSubsetDataset, RandomSubsetDataset 
from torchvision.models import resnet18, resnet50
import torch.nn as nn
import torch


class ContrastiveLearningDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, dataset_name, n_views):
        valid_datasets = {
            'sas': lambda: SASSubsetDataset(self.dataset,
                                            transform=ContrastiveLearningViewGenerator(
                                                self.get_simclr_pipeline_transform(32),
                                                n_views),
                                            ),

            'random': lambda: RandomSubsetDataset(self.dataset,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(96),
                                                    n_views),
                                                )
    }


        try:
            dataset_fn = valid_datasets[dataset_name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

class CIFAR100Augment(datasets.CIFAR100):
    def __init__(self, root: str, transform: callable, n_augmentations: int = 2, train: bool = True, download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, _ = self.data[index], self.targets[index]
        pil_img = Image.fromarray(img)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs

class PretrainedResnet(nn.Module):
    def __init__(self):
        super(PretrainedResnet, self).__init__()
        self.base_model = resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])  # Remove the last classification layer
        
    def forward(self, x1, ):
        with torch.no_grad():
            embedding_1 = self.base_model(x1)
        
        embedding_1_flat = torch.flatten(embedding_1, start_dim=1)
 
        return embedding_1_flat