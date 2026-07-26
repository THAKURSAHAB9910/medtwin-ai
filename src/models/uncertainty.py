import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict

class UncertaintyCalibrator(nn.Module):
    """
    Uncertainty Calibration and Conformal Prediction Engine.
    Provides temperature scaling for probability calibration and 
    split conformal prediction to guarantee error bounds with statistical coverage.
    """
    def __init__(self):
        super().__init__()
        # Learnable temperature scalar parameter (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.conformal_threshold = 0.5 # Default fallback

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrates logits by scaling them with temperature.
        """
        return logits / self.temperature

    def get_calibrated_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Applies temperature scaling and returns calibrated probabilities.
        """
        scaled_logits = self.forward(logits)
        return torch.sigmoid(scaled_logits)

    def fit_temperature(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, epochs: int = 100):
        """
        Optimizes the temperature parameter on validation data using negative log likelihood.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=epochs)
        
        # Binary cross entropy loss with logits
        bce_loss = nn.BCEWithLogitsLoss()
        
        val_logits = val_logits.detach()
        val_labels = val_labels.float().detach().view_as(val_logits)
        
        def eval_loss():
            optimizer.zero_grad()
            # Temperature must remain positive
            # Constrain temperature to be >= 0.01 to avoid division by zero
            with torch.no_grad():
                self.temperature.clamp_(min=0.01)
            loss = bce_loss(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        # Ensure final parameter clamp
        self.temperature.data.clamp_(min=0.01)
        print(f"Optimal Temperature Calibrated: {self.temperature.item():.4f}")

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes the Expected Calibration Error (ECE).
        ECE partitions predictions into M equally-spaced bins and computes the 
        weighted average difference between accuracy and confidence inside bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(probs)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in the current bin
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] >= 0.5))
                # For binary classification, confidence is max(p, 1-p)
                conf = np.where(probs[in_bin] >= 0.5, probs[in_bin], 1.0 - probs[in_bin])
                avg_confidence_in_bin = np.mean(conf)
                
                # Weighted absolute difference
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
        return float(ece)

    def fit_conformal(self, val_probs: np.ndarray, val_labels: np.ndarray, alpha: float = 0.1):
        """
        Calibrates Split Conformal Prediction thresholds.
        Args:
            val_probs: Calibrated probabilities of shape [N]
            val_labels: Ground truth binary labels of shape [N]
            alpha: Significance level (error rate tolerance). 
                   Guarantees that the true class is in the prediction set with probability >= 1 - alpha.
        """
        n = len(val_probs)
        
        # Calculate conformal non-conformity scores
        # For binary classification, conformity score is the probability of the true label
        # Non-conformity score s_i = 1 - P(Y_i | X_i)
        true_probs = np.where(val_labels == 1, val_probs, 1.0 - val_probs)
        nonconformity_scores = 1.0 - true_probs
        
        # Compute the (1 - alpha) quantile of nonconformity scores, adjusted for sample size
        # Formula: q = ceil((n + 1)(1 - alpha)) / n quantile
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        
        # Sort and pick the quantile value
        self.conformal_threshold = np.quantile(nonconformity_scores, q_level)
        print(f"Conformal Calibration Completed: threshold q = {self.conformal_threshold:.4f} (coverage: {1 - alpha:.2%})")

    def predict_conformal_set(self, probs: np.ndarray) -> List[List[int]]:
        """
        Constructs prediction sets for the given probabilities.
        Returns a list of prediction sets (e.g. [0], [1], or [0, 1] for uncertain).
        """
        prediction_sets = []
        for p in probs:
            p_set = []
            # Probability of Clean class (0) is 1 - p
            if (1.0 - p) >= (1.0 - self.conformal_threshold):
                p_set.append(0)
            # Probability of Tampered class (1) is p
            if p >= (1.0 - self.conformal_threshold):
                p_set.append(1)
                
            # Fallback if both are below threshold (should be empty, but we default to most likely)
            if len(p_set) == 0:
                p_set.append(int(p >= 0.5))
                
            prediction_sets.append(p_set)
        return prediction_sets

if __name__ == "__main__":
    # Test temperature optimization & ECE
    calibrator = UncertaintyCalibrator()
    
    # Mock data
    val_logits = torch.tensor([-2.5, -1.0, 0.5, 1.2, 2.5, 3.0])
    val_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    
    calibrator.fit_temperature(val_logits, val_labels, epochs=10)
    
    probs = calibrator.get_calibrated_probability(val_logits).detach().numpy()
    ece = calibrator.compute_ece(probs, val_labels.numpy())
    print(f"Calibrated probabilities: {probs}")
    print(f"ECE: {ece:.6f}")
    
    # Test conformal prediction
    calibrator.fit_conformal(probs, val_labels.numpy(), alpha=0.2)
    sets = calibrator.predict_conformal_set(probs)
    print(f"Conformal prediction sets: {sets}")
