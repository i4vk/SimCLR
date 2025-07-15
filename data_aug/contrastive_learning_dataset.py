from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from datasets.spectra_dataset import SpectraDataset
from data_aug.spectra_aug import *


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

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
    
    @staticmethod
    def get_spectra_simclr_pipeline_transform(s=1):
        """Return a set of data augmentation transformations for spectra data as described in the SimCLR paper."""
        # Transformación específica de espectros: tensorize y recorte
        augment = SpectralAugment(
            p_noise=0.5,
            p_scale=0.5,
            p_shift=0.5,
            p_savgol=0.2,
            p_derivative=0.0,
            p_crop=0.3,
            p_pca=0.0,
            p_emsa=0.0
        )
        def transform(x, wl):
            # Convertir array a tensor float
            x_t = torch.tensor(x, dtype=torch.float32)
            wl = torch.tensor(wl, dtype=torch.float32)
            # Aplicar augment que retorna (x_mod, wl_mod)
            return augment(x_t, wl)
        return transform

    # @staticmethod
    # def get_spectra_simclr_pipeline_transform(s=1):
    #     """Return a set of data augmentation transformations for spectra data as described in the SimCLR paper."""
    #     data_transforms = transforms.Compose([
    #         # Asegurarnos de que la señal es un float tensor
    #         transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    #         # Augmentación específica de espectros
    #         SpectralAugment(
    #             p_noise=0.5 * s,
    #             p_scale=0.5 * s,
    #             p_shift=0.5 * s,
    #             p_smooth=0.3 * s
    #         ),
    #     ])
    #     return data_transforms
        

    def get_dataset(self, name, n_views, split='supp'):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                            'spectra': lambda: SpectraDataset(self.root_folder, mode=split,
                                                            transform=ContrastiveLearningViewGenerator(
                                                                self.get_spectra_simclr_pipeline_transform(),
                                                                n_views))}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
