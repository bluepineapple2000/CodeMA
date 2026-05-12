import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from unet_model import UNet

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
dir_combined = Path('../data/combined_122/')
dir_ground_truth = Path('../data/jpg_122/')


class CombinedImageDataset(Dataset):
    """Dataset for loading combined images and their corresponding ground truth pairs."""
    
    def __init__(self, combined_dir, ground_truth_dir, img_scale=1.0):
        self.combined_dir = Path(combined_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.img_scale = img_scale
        
        # Get all combined image files
        self.combined_files = sorted([f for f in self.combined_dir.glob('*.png')])
        
        logging.info(f'Found {len(self.combined_files)} combined images')
    
    def parse_combined_filename(self, filename):
        """Parse combined filename to extract the two original image names.
        
        E.g., 'combined_snap_24_005_snap_24_006.png' -> ('snap_24_005.jpg', 'snap_24_006.jpg')
        """
        # Remove 'combined_' prefix and '.png' suffix
        base_name = filename.stem.replace('combined_', '')
        
        # Try all possible split points to find valid jpg files
        parts = base_name.split('_')
        
        for split_idx in range(1, len(parts)):
            name1 = '_'.join(parts[:split_idx])
            name2 = '_'.join(parts[split_idx:])
            
            file1 = self.ground_truth_dir / f'{name1}.jpg'
            file2 = self.ground_truth_dir / f'{name2}.jpg'
            
            if file1.exists() and file2.exists():
                return name1, name2, file1, file2
        
        raise ValueError(f"Could not parse filename {filename}")
    
    def __len__(self):
        return len(self.combined_files)
    
    def __getitem__(self, idx):
        combined_file = self.combined_files[idx]
        
        # Parse filename to get ground truth image paths
        name1, name2, gt_path1, gt_path2 = self.parse_combined_filename(combined_file)
        
        # Load combined image
        combined_img = Image.open(combined_file).convert('RGB')
        
        # Load ground truth images
        gt_img1 = Image.open(gt_path1).convert('RGB')
        gt_img2 = Image.open(gt_path2).convert('RGB')
        
        # Scale images
        if self.img_scale != 1.0:
            h, w = int(combined_img.size[1] * self.img_scale), int(combined_img.size[0] * self.img_scale)
            combined_img = combined_img.resize((w, h), Image.BILINEAR)
            gt_img1 = gt_img1.resize((w, h), Image.BILINEAR)
            gt_img2 = gt_img2.resize((w, h), Image.BILINEAR)
        
        # Convert to tensors
        combined_tensor = torch.from_numpy(np.array(combined_img)).float() / 255.0
        gt_tensor1 = torch.from_numpy(np.array(gt_img1)).float() / 255.0
        gt_tensor2 = torch.from_numpy(np.array(gt_img2)).float() / 255.0
        
        # Convert from HWC to CHW
        combined_tensor = combined_tensor.permute(2, 0, 1)
        gt_tensor1 = gt_tensor1.permute(2, 0, 1)
        gt_tensor2 = gt_tensor2.permute(2, 0, 1)
        
        # Convert RGB images to grayscale masks (using mean across channels)
        gt_mask1 = gt_tensor1.mean(dim=0, keepdim=True)
        gt_mask2 = gt_tensor2.mean(dim=0, keepdim=True)
        
        return {
            'image': combined_tensor,
            'gt_mask1': gt_mask1,
            'gt_mask2': gt_mask2,
            'combined_input': combined_tensor  # Keep original for crossroad loss
        }


def crossroad_l1_loss(mask1_pred, mask2_pred, gt_mask1, gt_mask2, combined_input, lambda_crossroad=1.0):
    """
    Crossroad L1 loss for image decomposition.
    
    Combines three losses:
    1. Direct reconstruction of mask1: L1(mask1_pred, gt_mask1)
    2. Direct reconstruction of mask2: L1(mask2_pred, gt_mask2)
    3. Crossroad loss: L1(mask1_pred + mask2_pred, combined_input_grayscale)
    
    Args:
        mask1_pred: Predicted first mask (B, 1, H, W)
        mask2_pred: Predicted second mask (B, 1, H, W)
        gt_mask1: Ground truth first mask (B, 1, H, W)
        gt_mask2: Ground truth second mask (B, 1, H, W)
        combined_input: Original combined input image (B, 3, H, W)
        lambda_crossroad: Weight for the crossroad loss
    
    Returns:
        Total loss
    """
    # Resize predictions to match ground truth shape if needed
    if mask1_pred.shape != gt_mask1.shape:
        mask1_pred = F.interpolate(mask1_pred, size=gt_mask1.shape[2:], mode='bilinear', align_corners=False)
    if mask2_pred.shape != gt_mask2.shape:
        mask2_pred = F.interpolate(mask2_pred, size=gt_mask2.shape[2:], mode='bilinear', align_corners=False)
    
    # Direct reconstruction losses
    loss_mask1 = F.l1_loss(mask1_pred, gt_mask1)
    loss_mask2 = F.l1_loss(mask2_pred, gt_mask2)
    
    # Convert combined input to grayscale for crossroad loss
    combined_gray = combined_input.mean(dim=1, keepdim=True)
    
    # Crossroad loss: combined masks should reconstruct the input
    reconstructed = mask1_pred + mask2_pred
    loss_crossroad = F.l1_loss(reconstructed, combined_gray)
    
    # Total loss
    total_loss = loss_mask1 + loss_mask2 + lambda_crossroad * loss_crossroad
    
    return total_loss


def evaluate(model, dataloader, device, amp):
    """
    Evaluate the model on validation set and return average Dice score.
    """
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                
                # Get ground truth masks
                gt_mask1 = batch['gt_mask1'].to(device=device, dtype=torch.float32)
                gt_mask2 = batch['gt_mask2'].to(device=device, dtype=torch.float32)
                
                # Split predictions into two masks
                mask1_pred = masks_pred[:, 0:1, :, :]
                mask2_pred = masks_pred[:, 1:2, :, :]
                
                # Apply sigmoid to get values in [0, 1]
                mask1_pred = torch.sigmoid(mask1_pred)
                mask2_pred = torch.sigmoid(mask2_pred)
                
                # Resize predictions to match ground truth if needed
                if mask1_pred.shape != gt_mask1.shape:
                    mask1_pred = F.interpolate(mask1_pred, size=gt_mask1.shape[2:], mode='bilinear', align_corners=False)
                if mask2_pred.shape != gt_mask2.shape:
                    mask2_pred = F.interpolate(mask2_pred, size=gt_mask2.shape[2:], mode='bilinear', align_corners=False)
                
                # Calculate Dice score for both masks
                dice1 = dice_coeff(mask1_pred, gt_mask1)
                dice2 = dice_coeff(mask2_pred, gt_mask2)
                dice_scores.append((dice1 + dice2) / 2.0)
    
    model.train()
    return torch.stack(dice_scores).mean().item()


def dice_coeff(input, target, smooth=1e-6):
    """
    Calculate Dice coefficient between input and target tensors.
    """
    intersection = (input * target).sum()
    union = input.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        lambda_crossroad: float = 1.0,
        use_combined_dataset: bool = True,
):

    if use_combined_dataset:
        dataset = CombinedImageDataset(dir_combined, dir_ground_truth, img_scale)
    else:
        raise ValueError("Non-combined dataset mode is not implemented. Please use --combined flag.")

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=device.type != 'cpu')
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    writer = SummaryWriter()
    writer.add_hparams(
        {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_percent': val_percent,
            'save_checkpoint': save_checkpoint,
            'img_scale': img_scale,
            'amp': amp
        },
        {'hparam/metric': 0}
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    if use_combined_dataset:
                        # Handle combined dataset with crossroad loss
                        gt_mask1 = batch['gt_mask1'].to(device=device, dtype=torch.float32)
                        gt_mask2 = batch['gt_mask2'].to(device=device, dtype=torch.float32)
                        combined_input = batch['combined_input'].to(device=device, dtype=torch.float32)
                        
                        # Split predictions into two masks
                        # Assuming model outputs 2 channels: [mask1_pred, mask2_pred]
                        mask1_pred = masks_pred[:, 0:1, :, :]  # First channel
                        mask2_pred = masks_pred[:, 1:2, :, :]  # Second channel
                        
                        # Apply sigmoid to get values in [0, 1]
                        mask1_pred = torch.sigmoid(mask1_pred)
                        mask2_pred = torch.sigmoid(mask2_pred)
                        
                        loss = crossroad_l1_loss(mask1_pred, mask2_pred, gt_mask1, gt_mask2, 
                                               combined_input, lambda_crossroad=lambda_crossroad)
                    else:
                        # Original training logic
                        true_masks = batch['mask'].to(device=device, dtype=torch.long)
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('train_loss', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                writer.add_histogram(f'Weights/{tag}', value.data, global_step)
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                writer.add_histogram(f'Gradients/{tag}', value.grad.data, global_step)

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                            writer.add_scalar('validation_Dice', val_score, global_step)
                            writer.add_image('input_images', images[0], global_step)
                            
                            if use_combined_dataset:
                                # Log predicted masks for combined dataset
                                writer.add_image('masks/pred_mask1', torch.sigmoid(masks_pred[0, 0:1, :, :]).float(), global_step)
                                writer.add_image('masks/pred_mask2', torch.sigmoid(masks_pred[0, 1:2, :, :]).float(), global_step)
                            else:
                                # Original logging for standard dataset
                                writer.add_image('masks/true', true_masks[0].float().unsqueeze(0), global_step)
                                writer.add_image('masks/pred', masks_pred.argmax(dim=1)[0].float().unsqueeze(0), global_step)
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--combined', action='store_true', default=False, 
                        help='Use combined dataset with crossroad loss for image decomposition')
    parser.add_argument('--lambda-crossroad', type=float, default=1.0,
                        help='Weight for crossroad loss in combined dataset mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # For combined dataset: output 2 channels (one per mask)
    n_classes = 2 if args.combined else args.classes
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_combined_dataset=args.combined,
            lambda_crossroad=args.lambda_crossroad
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_combined_dataset=args.combined,
            lambda_crossroad=args.lambda_crossroad
        )