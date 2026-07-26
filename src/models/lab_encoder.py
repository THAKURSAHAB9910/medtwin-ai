import torch
import torch.nn as nn

class StructuredLabEncoder(nn.Module):
    """
    Structured Clinical Vitals & Lab Reports Encoder.
    Processes tabular clinical parameters (e.g. vitals, blood panels) and projects
    them into a shared semantic embedding token.
    Vector structure: [age, temperature, heart_rate, wbc_count, oxygen_saturation]
    """
    def __init__(self, input_dim: int = 5, embed_dim: int = 256):
        super().__init__()
        # Normalization layer for medical vital metrics (to avoid scale dominance)
        self.input_layer = nn.BatchNorm1d(input_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input vitals tensor [B, input_dim]
        Returns:
            lab_token: Embeddings representing the vital state [B, 1, embed_dim]
        """
        # Apply batch normalization (needs float)
        normalized = self.input_layer(x.float())
        out = self.mlp(normalized)
        # Add singleton dimension for sequence representation [B, 1, embed_dim]
        return out.unsqueeze(1)

if __name__ == "__main__":
    encoder = StructuredLabEncoder()
    mock_labs = torch.tensor([[45.0, 98.6, 75.0, 6000.0, 98.0], [67.0, 102.5, 110.0, 15000.0, 90.0]])
    token = encoder(mock_labs)
    print("Structured Lab Encoder dimensions:")
    print(f" - Out token shape: {token.shape}")
