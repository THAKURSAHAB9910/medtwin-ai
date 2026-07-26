import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple

class BayesianPatientTwinModel(nn.Module):
    """
    Bayesian neural network simulating state-space updates.
    State representation St: [weight (kg), blood_pressure (mmHg), blood_glucose (mg/dL), heart_rate (bpm), wbc_count (uL), SpO2 (%)]
    Action representation At: [weight_change (kg), metformin_dose (mg), beta_blocker_dose (mg), exercise_hrs (hrs/wk), sodium_intake (g)]
    
    Uses Monte Carlo (MC) Dropout to estimate epistemic uncertainty over future predictions.
    """
    def __init__(self, state_dim: int = 6, action_dim: int = 5, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # Simple transition MLP: projects [St || At] -> St+1 delta
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates state delta: St+1 = St + delta
        Forces dropout active even during model.eval() to support MC Dropout.
        """
        x = torch.cat([state, action], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=True) # MC Dropout active
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_rate, training=True)
        
        delta = self.fc3(x)
        
        # Bounds delta updates to prevent extreme divergence during long rollouts
        delta = torch.tanh(delta) * 15.0 # Max vital delta of 15 units per time step
        
        return state + delta

    def rollout_simulation(
        self, 
        initial_state: torch.Tensor, 
        actions: torch.Tensor, 
        steps: int = 12, 
        num_mc_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs recursive step predictions with Monte Carlo samples to estimate trajectory uncertainty.
        initial_state shape: [1, state_dim]
        actions shape: [steps, action_dim]
        
        Returns:
            means: [steps + 1, state_dim]
            stds:  [steps + 1, state_dim]
        """
        self.eval()
        all_samples = []
        
        for _ in range(num_mc_samples):
            trajectory = [initial_state.clone()]
            current_state = initial_state.clone()
            
            for t in range(steps):
                action_t = actions[t:t+1] # [1, action_dim]
                with torch.no_grad():
                    # Forward pass updates state
                    current_state = self.forward(current_state, action_t)
                trajectory.append(current_state.clone())
                
            # Stack to [steps + 1, state_dim]
            all_samples.append(torch.cat(trajectory, dim=0).cpu().numpy())
            
        all_samples = np.array(all_samples) # [num_mc_samples, steps + 1, state_dim]
        
        means = np.mean(all_samples, axis=0)
        stds = np.std(all_samples, axis=0)
        
        return means, stds

if __name__ == "__main__":
    model = BayesianPatientTwinModel()
    
    # Mock initial state: Weight=85kg, BP=135mmHg, Glucose=140mg/dL, HR=80bpm, WBC=7500, SpO2=98%
    s0 = torch.tensor([[85.0, 135.0, 140.0, 80.0, 7500.0, 98.0]], dtype=torch.float32)
    
    # Mock actions for 12 months: Weight loss of 0.8kg/mo, 1000mg Metformin, 0mg BetaBlocker, 3h exercise/wk, 2.0g Sodium
    a = torch.tensor([[-0.8, 1000.0, 0.0, 3.0, 2.0]] * 12, dtype=torch.float32)
    
    means, stds = model.rollout_simulation(s0, a, steps=12, num_mc_samples=30)
    print("Bayesian Patient Twin rollout completed:")
    print(f" - Final projected Weight:  {means[-1, 0]:.2f} +/- {stds[-1, 0]:.2f} kg")
    print(f" - Final projected Glucose: {means[-1, 2]:.2f} +/- {stds[-1, 2]:.2f} mg/dL")
