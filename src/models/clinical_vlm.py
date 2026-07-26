import torch
import torch.nn as nn
from typing import Tuple, List
from transformers import AutoModel, AutoTokenizer

class ClinicalVLM(nn.Module):
    """
    Clinical text processing branch of MedTwin AI.
    Processes clinical notes, patient transcriptions, and medical reports.
    Integrates with Hugging Face clinical language models (e.g. ClinicalBERT) or falls back to custom transformer.
    """
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 256,
        use_mock: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.use_mock = use_mock
        
        if not self.use_mock:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.encoder = AutoModel.from_pretrained(model_name)
                hidden_size = getattr(self.encoder.config, "hidden_size", 768)
                self.projection = nn.Linear(hidden_size, embed_dim)
            except Exception as e:
                print(f"Warning: Failed to load clinical model ({e}). Falling back to mock text encoder.")
                self.use_mock = True
                
        if self.use_mock:
            self.vocab_size = 8000
            self.embeddings = nn.Embedding(self.vocab_size, 64)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.projection = nn.Linear(64, embed_dim)

    def _mock_tokenize(self, texts: List[str], max_len: int = 64, device: torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        
        for i, text in enumerate(texts):
            words = text.split()
            for j, word in enumerate(words[:max_len]):
                token_id = (hash(word) % (self.vocab_size - 2)) + 1
                input_ids[i, j] = token_id
                attention_mask[i, j] = 1.0
        return input_ids, attention_mask

    def forward(self, texts: List[str], device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            texts: List of patient clinical note strings.
        Returns:
            features: Semantic token features [B, seq_len, embed_dim]
            attention_mask: Attention masks [B, seq_len]
        """
        if not self.use_mock:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs)
            features = self.projection(outputs.last_hidden_state)
            return features, inputs["attention_mask"]
        else:
            ids, mask = self._mock_tokenize(texts, max_len=64, device=device)
            embeds = self.embeddings(ids)
            encoded = self.transformer(embeds, src_key_padding_mask=(mask == 0))
            features = self.projection(encoded)
            return features, mask

if __name__ == "__main__":
    vlm = ClinicalVLM(use_mock=True)
    feats, mask = vlm(["Patient coughs intensely.", "Clear pulmonary fields."])
    print("Clinical VLM Text dimensions:")
    print(f" - Features: {feats.shape}")
    print(f" - Mask: {mask.shape}")
