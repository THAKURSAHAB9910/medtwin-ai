import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.generator import PatientCraftEngine

class MedicalMultimodalDataset(Dataset):
    """
    Multimodal medical clinical dataset.
    Loads chest radiographs, segmentations, textual clinical notes, 
    and structured clinical vitals from PatientCraftEngine.
    """
    def __init__(
        self,
        num_samples: int = 100,
        is_training: bool = True,
        pneumonia_prob: float = 0.5,
        noise_level: float = 0.2,
        image_size: Tuple[int, int] = (384, 384),
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.is_training = is_training
        self.pneumonia_prob = pneumonia_prob
        self.noise_level = noise_level
        self.image_size = image_size
        
        # Initialize simulation engine
        self.engine = PatientCraftEngine(seed=seed if seed is not None else (42 if is_training else 100))
        
        # Normalization metrics for ImageNet
        if self.is_training:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                # Medical scans can have light rotation or contrast changes
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
        # Distribute labels
        self.labels = [1 if i < int(num_samples * pneumonia_prob) else 0 for i in range(num_samples)]
        np.random.shuffle(self.labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        has_pneumonia = self.labels[idx] == 1
        
        sample = self.engine.generate_patient_record(has_pneumonia=has_pneumonia, noise_level=self.noise_level)
        
        img = np.array(sample["image"])
        mask = np.array(sample["mask"])
        
        # Resize mask to target image dimensions
        pil_mask = Image.fromarray(mask).resize(self.image_size, Image.Resampling.NEAREST)
        mask_resized = np.array(pil_mask, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Apply visual augmentations
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]
        mask_tensor = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0)
        
        # Format structured lab metrics into a tensor vector:
        # Vector features: [age, temperature, heart_rate, wbc_count, oxygen_saturation]
        labs = sample["lab_metrics"]
        lab_vector = torch.tensor([
            labs["age"],
            labs["temperature"],
            labs["heart_rate"],
            labs["wbc_count"],
            labs["oxygen_saturation"]
        ], dtype=torch.float32)
        
        return {
            "image": img_tensor,              # [3, H, W]
            "mask": mask_tensor,              # [1, H, W]
            "clinical_note": sample["clinical_note"], # Physician transcription string
            "lab_metrics": lab_vector,        # [5] structured vitals
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "metadata": sample["metadata"]     # Raw clinical vitals markers
        }


def custom_collate_fn(batch):
    """
    Custom collator mapping lists and dictionaries without breaking DataLoader.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    lab_metrics = torch.stack([item["lab_metrics"] for item in batch], dim=0)
    
    clinical_notes = [item["clinical_note"] for item in batch]
    metadata = [item["metadata"] for item in batch]
    
    return {
        "image": images,
        "mask": masks,
        "clinical_note": clinical_notes,
        "lab_metrics": lab_metrics,
        "label": labels,
        "metadata": metadata
    }


class MedicalDataModule(pl.LightningDataModule):
    """
    LightningDataModule managing train, validation, and OOD test splits.
    """
    def __init__(
        self,
        batch_size: int = 4,
        train_samples: int = 40,
        val_samples: int = 15,
        test_samples: int = 15,
        image_size: Tuple[int, int] = (384, 384),
        num_workers: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.image_size = image_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MedicalMultimodalDataset(
            num_samples=self.train_samples,
            is_training=True,
            pneumonia_prob=0.5,
            noise_level=0.1, # Normal noise
            image_size=self.image_size,
            seed=42
        )
        
        self.val_dataset = MedicalMultimodalDataset(
            num_samples=self.val_samples,
            is_training=False,
            pneumonia_prob=0.5,
            noise_level=0.1,
            image_size=self.image_size,
            seed=100
        )
        
        # Out-Of-Distribution (OOD) test set: high scan noise simulating degraded field scans
        self.test_dataset = MedicalMultimodalDataset(
            num_samples=self.test_samples,
            is_training=False,
            pneumonia_prob=0.5,
            noise_level=0.5, # Severe noise
            image_size=self.image_size,
            seed=200
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )

if __name__ == "__main__":
    dm = MedicalDataModule(batch_size=2)
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    print("Dataset DataLoader batch structures:")
    print(f" - Image shape: {batch['image'].shape}")
    print(f" - Lab metrics shape: {batch['lab_metrics'].shape}")
    print(f" - Clinical Note count: {len(batch['clinical_note'])}")
    print(f" - Metadata count: {len(batch['metadata'])}")
