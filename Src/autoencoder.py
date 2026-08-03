"""
Convolutional Autoencoder for PCB Anomaly Detection.

Concept:
- Trained ONLY on clean/normal PCB images.
- Learns to compress (encode) and reconstruct (decode) normal PCB patterns.
- Never sees defects during training.
- At inference: feed any PCB image through the model.
    - Normal PCB  -> low reconstruction error (model has seen this pattern before)
    - Defective PCB -> high reconstruction error (model has NOT seen this pattern,
      so it reconstructs it poorly, especially around the defect region)
- The pixel-wise difference between input and reconstruction gives a heatmap
  that highlights *where* the anomaly is, even without ever being trained on
  labeled defects.

This complements YOLOv8: YOLO catches known, labeled defect types fast.
The autoencoder catches novel/rare anomalies that don't match any training class.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Symmetric convolutional autoencoder.
    Input: (B, 3, 256, 256) normalized PCB image crop.
    Output: (B, 3, 256, 256) reconstruction.

    Architecture rationale:
    - Encoder progressively downsamples spatial size while increasing channel
      depth, forcing the network to learn a compressed latent representation
      of "what a normal PCB trace/pad/silkscreen pattern looks like".
    - Decoder mirrors the encoder with transposed convolutions to upsample
      back to the original resolution.
    - BatchNorm + ReLU stabilize training; final Sigmoid keeps output in [0,1]
      to match normalized input range.
    """

    def __init__(self, latent_channels: int = 256):
        super().__init__()

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 256 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


def reconstruction_error_map(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Compute a per-pixel anomaly heatmap.

    We average the absolute difference across the 3 color channels, giving
    a single-channel (B, 1, H, W) heatmap where high values = likely defect.
    """
    diff = torch.abs(original - reconstructed)
    heatmap = diff.mean(dim=1, keepdim=True)
    return heatmap


def anomaly_score(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Scalar anomaly score per image = mean reconstruction error over all pixels.
    Used to set a pass/fail threshold (e.g. calibrated from validation set
    of known-clean PCBs: threshold = mean + 3*std of clean reconstruction error).
    """
    diff = torch.abs(original - reconstructed)
    return diff.mean(dim=[1, 2, 3])
