import torch
import json
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List

class ClinicalSFTDataset(Dataset):
    """
    Formats Patient Records into Supervised Fine-Tuning (SFT) input/label pairs.
    Masks prompt tokens with -100 to calculate cross-entropy loss exclusively
    on the target diagnostic response.
    """
    def __init__(self, patient_records: List[Dict[str, Any]], tokenizer: Any, max_length: int = 256):
        self.records = patient_records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        note = record.get("clinical_note", "")
        vitals = record.get("lab_metrics", {})
        label = record.get("label", 0) # 0 = Normal, 1 = Pneumonia
        
        diag_str = "Pneumonia" if label == 1 else "Normal"
        
        # 1. Format prompt
        vitals_str = json.dumps(vitals)
        prompt = f"CLINICAL NOTE: {note}\nVITALS: {vitals_str}\nDIAGNOSTIC RECOMMENDATION: "
        target = f"{diag_str}."
        
        # 2. Tokenize prompt & target
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        
        input_ids = prompt_ids + target_ids
        
        # Crop to max length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            
        # 3. Create labels matrix (-100 masks loss computation on prompts)
        labels = [-100] * len(prompt_ids) + target_ids
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
            
        # Pad sequence to max length
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len
            
        # 4. Mock pixel representation (if scan is present)
        # SOTA models like Qwen2-VL require pixel values aligned to spatial grids
        pixel_values = torch.randn(3, 224, 224)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values
        }

class MockTokenizer:
    """
    Mock Tokenizer supporting pad, encode, and decode configurations.
    """
    def __init__(self):
        self.pad_token_id = 0
        self.vocab = {"Normal": 10, "Pneumonia": 11, ".": 12, "Normal.": 13, "Pneumonia.": 14}
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        # Basic mapping helper
        words = text.split()
        ids = [hash(w) % 1000 + 20 for w in words]
        if "Normal." in text:
            ids.append(13)
        elif "Pneumonia." in text:
            ids.append(14)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        # Check if targets match vocabulary
        if 13 in token_ids or 14 in token_ids:
            return "Pneumonia." if 14 in token_ids else "Normal."
        return "Normal."

def create_sft_dataloader(records: List[Dict[str, Any]], tokenizer: Any, batch_size: int = 4) -> DataLoader:
    dataset = ClinicalSFTDataset(records, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
