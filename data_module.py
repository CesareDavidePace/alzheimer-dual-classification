import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms, datasets
import os
import yaml
from PIL import Image

class PairedAlzheimerDataset(Dataset):
    def __init__(self, root_dir, fold="Fold1", split="Train", transform_offline=None, transform_rgb=None, split_division=None):
        """
        Args:
            root_dir (str): Root directory containing the dataset.
            fold (str): Current fold to use.
            split (str): Split to use (Train, Val, Test).
            transform_offline (callable, optional): Transformations for offline images.
            transform_rgb (callable, optional): Transformations for rgb images.
            split_division (dict): Split division loaded from the YAML file.
        """
        self.root_dir = root_dir
        self.fold = fold
        self.split = split
        self.transform_offline = transform_offline
        self.transform_rgb = transform_rgb
        self.split_division = split_division

        # Ensure split division is provided
        if self.split_division is None:
            raise ValueError("split_division cannot be None. Please provide the YAML split division.")

        # Extract file names for the given fold and split
        self.file_names = self.split_division[self.fold][self.split]

        # Initialize lists to store image paths
        self.offline_images = []
        self.rgb_images = []

       # Collect images based on file names
        for file_name in self.file_names:
            offline_image_path = os.path.join(self.root_dir, "offline", "TASK_08", file_name)
            rgb_image_path = os.path.join(self.root_dir, "rgb_in_air_on_paper", "TASK_08", file_name)

            if os.path.exists(offline_image_path) and os.path.exists(rgb_image_path):
                self.offline_images.append(offline_image_path)
                self.rgb_images.append(rgb_image_path)
            else:
                # Skip the file if either image is not found
                # print(f"Skipping {file_name}: offline or RGB image not found.")
                continue


        # print the number of items in the dataset
        print(f"Number of {self.split} items in {self.fold}: {len(self.offline_images)}")

        # Ensure paired lists have the same length
        assert len(self.offline_images) == len(self.rgb_images), "Mismatch in offline and RGB image counts."

    def __len__(self):
        return len(self.offline_images)

    def __getitem__(self, idx):
        # Load offline and RGB images
        offline_img = Image.open(self.offline_images[idx]).convert("L")
        rgb_img = Image.open(self.rgb_images[idx]).convert("RGB")

        # Apply transformations
        if self.transform_offline:
            offline_img = self.transform_offline(offline_img)
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)

        # Extract label from the file name (e.g., HC or PT from `E06_T08_HC6_055`)
        label_name = self.offline_images[idx].split("_")[3][:2]
        label = 0 if label_name == "HC" else 1  # Assuming binary classification: HC=0, PT=1

        # Extract task from the file name (e.g., T08 from `E06_T08_HC6_055`)
        task = self.offline_images[idx].split("_")[2]

        return {"rgb": rgb_img, "single_channel": offline_img, "label": label, "task": task}


class AlzheimerDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, yaml_path, batch_size=32, num_workers=4, fold="Fold1"):
        """
        Args:
            root_dir (str): Root directory containing the dataset.
            yaml_path (str): Path to the YAML file containing split division.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.root_dir = root_dir
        self.yaml_path = yaml_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold

        # Load split division from YAML file
        with open(self.yaml_path, "r") as file:
            self.split_division = yaml.safe_load(file)

        # Define transforms
        self.transform_rgb = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_offline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.456], std=[0.224])
        ])

    def _print_split(self, split_name, dataset):
        """
        Print details of a dataset split.
        Args:
            split_name (str): Name of the split (e.g., Train, Val, Test).
            dataset (Dataset): The dataset to inspect.
        """
        print(f"--- {split_name} Split ---")
        for idx, file_name in enumerate(dataset.file_names):
            print(f"{split_name} item {idx}: {file_name}")
        print(f"Total {split_name} items: {len(dataset)}")

    def setup(self, stage=None):
        # Now self.fold retains the value set during initialization
        if stage == "fit" or stage is None:
            self.train_dataset = PairedAlzheimerDataset(
                root_dir=self.root_dir,
                fold=self.fold,
                split="Train",
                transform_offline=self.transform_offline,
                transform_rgb=self.transform_rgb,
                split_division=self.split_division,
            )
            self.val_dataset = PairedAlzheimerDataset(
                root_dir=self.root_dir,
                fold=self.fold,
                split="Val",
                transform_offline=self.transform_offline,
                transform_rgb=self.transform_rgb,
                split_division=self.split_division,
            )
        if stage == "test" or stage is None:
            self.test_dataset = PairedAlzheimerDataset(
                root_dir=self.root_dir,
                fold=self.fold,
                split="Test",
                transform_offline=self.transform_offline,
                transform_rgb=self.transform_rgb,
                split_division=self.split_division,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
