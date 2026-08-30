import torch
import torch.nn as nn

class ASLGestureClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(ASLGestureClassifier, self).__init__()
        # Input takes flattened summary features: [Mean (258), Delta/Velocity (258), Std (258)] = 774
        summary_input_dim = input_size * 3
        
        self.net = nn.Sequential(
            nn.Linear(summary_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 30, 258)
        mean_feat = torch.mean(x, dim=1)
        delta_feat = x[:, -1, :] - x[:, 0, :]
        std_feat = torch.std(x, dim=1)

        combined = torch.cat([mean_feat, delta_feat, std_feat], dim=1)
        return self.net(combined)