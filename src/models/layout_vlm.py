import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer

class LayoutVLM(nn.Module):
    """
    Layout VLM wrapper for semantic document understanding.
    Processes serialized OCR values, layout bounds, and structural text.
    Binds textual semantics to spatial layouts.
    """
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        embed_dim: int = 256,
        use_mock: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.use_mock = use_mock
        
        if not self.use_mock:
            try:
                # Load real Hugging Face layout VLM or tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.vlm = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                # Linear layer to map VLM hidden dimension to shared embedding space
                hidden_size = getattr(self.vlm.config, "hidden_size", 768)
                self.projection = nn.Linear(hidden_size, embed_dim)
            except Exception as e:
                print(f"Warning: Failed to load real VLM ({e}). Falling back to mock VLM.")
                self.use_mock = True
                
        if self.use_mock:
            # Mock VLM implementation: Lightweight Text Transformer Encoder
            # Maps vocabulary to hidden state, then simulates text representation
            self.vocab_size = 5000
            self.word_embeddings = nn.Embedding(self.vocab_size, 64)
            
            # Simple Transformer Encoder Layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            # Project to shared embedding space
            self.projection = nn.Linear(64, embed_dim)
            
    def _mock_tokenize(self, texts: list, max_len: int = 128, device: torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple deterministic tokenization for the mock model based on character hashing.
        """
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        
        for i, text in enumerate(texts):
            words = text.split()
            for j, word in enumerate(words[:max_len]):
                # Hash word to vocab index
                token_id = (hash(word) % (self.vocab_size - 2)) + 1
                input_ids[i, j] = token_id
                attention_mask[i, j] = 1.0
                
        return input_ids, attention_mask

    def forward(self, texts: list, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes text queries/OCR layouts.
        Args:
            texts: List of strings containing serialized document fields.
            device: Device to place tensors on.
        Returns:
            semantic_features: Encoded token embeddings [B, seq_len, embed_dim]
            attention_mask: Attention mask [B, seq_len]
        """
        if not self.use_mock:
            # Real VLM forward pass
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.vlm(**inputs)
            # Take last hidden states
            last_hidden = outputs.last_hidden_state
            semantic_features = self.projection(last_hidden)
            return semantic_features, inputs["attention_mask"]
        else:
            # Mock forward pass
            input_ids, attention_mask = self._mock_tokenize(texts, max_len=64, device=device)
            embeddings = self.word_embeddings(input_ids)
            encoded = self.transformer(embeddings, src_key_padding_mask=(attention_mask == 0))
            semantic_features = self.projection(encoded)
            return semantic_features, attention_mask

if __name__ == "__main__":
    # Validate tensor flows
    vlm = LayoutVLM(use_mock=True, embed_dim=256)
    sample_texts = [
        "Account Holder: John Doe | Beginning Balance: $10,000.00 |",
        "Account Holder: Alice Smith | Ending Balance: $25,000.00 |"
    ]
    features, mask = vlm(sample_texts)
    print("Layout VLM Branch Check:")
    print(f"Features shape: {features.shape} (Expected: [2, 64, 256])")
    print(f"Mask shape: {mask.shape} (Expected: [2, 64])")
