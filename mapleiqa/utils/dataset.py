from torch.utils.data import Dataset
from torchvision import transforms
from torch import tensor
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

GLOBAL_TRANSFORM_LIST = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )
            ]
    
class CustomKonIQ10kDataset(Dataset):
    """Custom KonIQ10k dataset."""

    def __init__(self, img_data, mos_value, device = "cpu", mos_norm=5, train_set=True, transform_list=None):
        """
        Arguments:
            img_data (array): array of path to the images of dataset.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_data = img_data
        self.mos_value = mos_value
        self.device = device
        self.mos_norm = mos_norm

        if transform_list is None:
            transform_list = GLOBAL_TRANSFORM_LIST
            if train_set:
                transform_list.extend([transforms.RandomHorizontalFlip()])
                
        self.data_transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.img_data)

    def load_and_transform_vision_data(self, image_path):
        if image_path is None:
            return None

        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            image = self.data_transform(image).to(self.device)

        return image

    def __getitem__(self, idx):

        img = self.load_and_transform_vision_data(self.img_data[idx])
        return img, (self.mos_value[idx]/self.mos_norm).to(self.device)
    
class CustomTIDDataset(Dataset):
    """Custom TID dataset."""

    def __init__(self, img_data, mos_value, device = "cpu", mos_norm=9, train_set=True, transform_list=None):
        """
        Arguments:
            img_data (array): array of path to the images of dataset.
            mos_value (string): array of MOS of dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_data = img_data
        self.mos_value = mos_value
        self.device = device
        self.mos_norm = mos_norm

        if transform_list is None:
            transform_list = GLOBAL_TRANSFORM_LIST
            if train_set:
                transform_list.extend([transforms.RandomHorizontalFlip()])
                
        self.data_transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.img_data)

    def load_and_transform_vision_data(self, image_path):
        if image_path is None:
            return None

        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            image = self.data_transform(image).to(self.device)

        return image

    def __getitem__(self, idx):

        img = self.load_and_transform_vision_data(self.img_data[idx])
        return img, tensor(self.mos_value[idx]/self.mos_norm).float().to(self.device)
        
class CustomSPAQDataset(Dataset):
    """Custom SPAQ dataset."""

    def __init__(self, img_data, mos_value, device = "cpu", mos_norm=100, train_set=True, transform_list=None):
        """
        Arguments:
            img_data (array): array of path to the images of dataset.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_data = img_data
        self.mos_value = mos_value
        self.device = device
        self.mos_norm = mos_norm

        if transform_list is None:
            transform_list = GLOBAL_TRANSFORM_LIST
            if train_set:
                transform_list.extend([transforms.RandomHorizontalFlip()])
                
        self.data_transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.img_data)

    def load_and_transform_vision_data(self, image_path):
        if image_path is None:
            return None

        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            image = self.data_transform(image).to(self.device)

        return image

    def __getitem__(self, idx):

        img = self.load_and_transform_vision_data(self.img_data[idx])
        return img, (self.mos_value[idx]/self.mos_norm).to(self.device)
    
class CustomLIVEDataset(Dataset):
    """Custom LIVE dataset."""

    def __init__(self, img_data, mos_value, device = "cpu", mos_norm=100, train_set=True, transform_list=None):
        """
        Arguments:
            img_data (array): array of path to the images of dataset.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_data = img_data
        self.mos_value = mos_value
        self.device = device
        self.mos_norm = mos_norm

        if transform_list is None:
            transform_list = GLOBAL_TRANSFORM_LIST
            if train_set:
                transform_list.extend([transforms.RandomHorizontalFlip()])
                
        self.data_transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.img_data)

    def load_and_transform_vision_data(self, image_path):
        if image_path is None:
            return None

        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            image = self.data_transform(image).to(self.device)

        return image

    def __getitem__(self, idx):

        img = self.load_and_transform_vision_data(self.img_data[idx])
        return img, (self.mos_value[idx]/self.mos_norm).to(self.device)
    
class CustomKADID10kDataset(Dataset):
    """Custom KADID10k dataset."""

    def __init__(self, img_data, mos_value, device = "cpu", mos_norm=5, train_set=True, transform_list=None):
        """
        Arguments:
            img_data (array): array of path to the images of dataset.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_data = img_data
        self.mos_value = mos_value
        self.device = device
        self.mos_norm = mos_norm

        if transform_list is None:
            transform_list = GLOBAL_TRANSFORM_LIST
            if train_set:
                transform_list.extend([transforms.RandomHorizontalFlip()])
                
        self.data_transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.img_data)

    def load_and_transform_vision_data(self, image_path):
        if image_path is None:
            return None

        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            image = self.data_transform(image).to(self.device)

        return image

    def __getitem__(self, idx):

        img = self.load_and_transform_vision_data(self.img_data[idx])
        return img, (self.mos_value[idx]/self.mos_norm).to(self.device)