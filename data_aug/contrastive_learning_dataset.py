from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
# Added for SAS evaluation
from sas.subset_dataset import SASSubsetDataset, RandomSubsetDataset 


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
            'sas': lambda: SASSubsetDataset(self.dataset, train=True,
                                            transform=ContrastiveLearningViewGenerator(
                                                self.get_simclr_pipeline_transform(32),
                                                n_views),
                                            ),

            'random': lambda: RandomSubsetDataset(self.dataset, train=True,
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
