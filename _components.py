class BraTSDataset(Dataset):
    """
    PyTorch Dataset for loading 2D BraTS slices stored as .h5 files.

    Each file contains:
        - image: (H, W, 4)
              MRI modalities in order: [T1, T1Gd, T2, T2-FLAIR]
        - mask:  (H, W, 3)
              Binary tumor subregions: [NEC/NET, ED, ET]

    Output format:
        - image tensor: (4, H, W), float32
        - mask tensor:  (3, H, W), float32
    """

    def __init__(self, slice_paths):
        """
        Args:
            slice_paths (list): List of file paths to .h5 slice files.
        """
        self.slice_paths = slice_paths

    def __len__(self):
        """
        Returns total number of slices and enables use with PyTorch DataLoader.
        """
        return len(self.slice_paths)

    def __getitem__(self, idx):
        """
        Loads and processes a single slice from disk.

        Steps:
            1. Load image + mask from disk (lazy loading)
            2. Convert to float32 (memory + GPU compatible)
            3. Apply per-slice, per-modality z-score normalization
            3. Rearrange to channel-first format (C, H, W) for PyTorch
            4. Convert to torch.Tensor
        """

        file_path = self.slice_paths[idx]

        # Load slice from disk (lazy loading; no full dataset in memory)
        with h5py.File(file_path, 'r') as file:
            image = file['image'][:]   # Shape: (H, W, 4)
            mask  = file['mask'][:]    # Shape: (H, W, 3)

        # Ensure correct dtype for training (convert to float32)
        image = image.astype(np.float32)
        mask  = mask.astype(np.float32)  # Also converted for loss computation; Binary values for each tumor subregion

        # Per-slice & per-modality z-score normalization to standardize intensity dist. independently for each MRI modality channel
        for c in range(image.shape[-1]):
            channel = image[:, :, c]
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                image[:, :, c] = (channel - mean) / std  # Improves optimization stability and handles inter-patient intensity variation
            else:
                image[:, :, c] = channel - mean  # Avoid division by zero if std=0

        # Convert to channel-first format (C, H, W) (required for PyTorch Conv2D layers)
        image = np.transpose(image, (2, 0, 1))  # (4, H, W)
        mask  = np.transpose(mask, (2, 0, 1))   # (3, H, W)

        # Convert numpy arrays to torch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor  = torch.from_numpy(mask)

        return image_tensor, mask_tensor


class LeanUNet(nn.Module):
    """
    U-Net (v1) for multi-class 2D brain tumor segmentation.

    Input: 4-channel MRI slice (T1, T1Gd, T2, T2-FLAIR)
    Output: 3-channel segmentation mask (Necrotic Core, Edema, Enhancing Tumor)
    """


    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        super(LeanUNet, self).__init__()
        features = init_features

        ### Encoder (downsampling path)
        self.encoder1 = self._conv_block(in_channels, features)
        self.pool1    = nn.MaxPool2d(2)

        self.encoder2 = self._conv_block(features, features*2)
        self.pool2    = nn.MaxPool2d(2)

        self.encoder3 = self._conv_block(features*2, features*4)
        self.pool3    = nn.MaxPool2d(2)

        ### Bottleneck
        self.bottleneck = self._conv_block(features*4, features*8)

        #### Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(features*8, features*4)

        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(features*4, features*2)

        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(features*2, features)

        ### Output layer
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)


    def forward(self, x):
        ### Encoder forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        ### Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        ### Decoder forward pass with skip connections
        dec3 = self.upconv3(bottleneck)  # upsample
        dec3 = torch.cat((dec3, enc3), dim=1)  # concatenate skip connection from encoder
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        ### Output
        return self.conv_final(dec1)


    ### 2-conv block
    def _conv_block(self, in_channels, out_channels):
        """
        Standard double-convolution block:
            Conv2d -> ReLU -> Conv2d -> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
