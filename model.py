import torch
import torch.nn as nn

class MathDeptCNN(nn.Module):
    def __init__(self):
        super(MathDeptCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        if x.dim() == 4: x = x.squeeze(2)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)