"""Debug script to identify training bottlenecks."""
import torch
import torch.nn as nn
import os
import sys

# The check/cross glyphs below are UTF-8; the default Windows console is cp1252
# and would raise UnicodeEncodeError on every status print. Force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from app import (
    MultiTaskUNet, DiceBCELoss, dice_coefficient, SegmentationDataset,
    collect_data, device
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def test_model_forward():
    """Test if model forward pass works without hanging."""
    print("Testing model forward pass...")
    model = MultiTaskUNet(in_channels=3, num_classes=4).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    dummy_masks = torch.randn(2, 4, 224, 224).to(device)

    try:
        with torch.no_grad():
            cls_logits, pred_masks = model(dummy_input)
            print(f"✓ Forward pass OK. cls_logits shape: {cls_logits.shape}, pred_masks shape: {pred_masks.shape}")
            return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_loss_computation():
    """Test if loss computation hangs or produces NaN."""
    print("\nTesting loss computation...")

    class_crit = nn.CrossEntropyLoss()
    seg_crit = DiceBCELoss()

    dummy_cls = torch.randn(2, 4).to(device)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long).to(device)
    dummy_masks = torch.rand(2, 4, 224, 224).to(device)
    dummy_targets = torch.rand(2, 4, 224, 224).to(device)

    try:
        class_loss = class_crit(dummy_cls, dummy_labels)
        print(f"  Class loss: {class_loss.item():.4f}")

        seg_loss = seg_crit(dummy_masks, dummy_targets)
        print(f"  Seg loss: {seg_loss.item():.4f}")

        total_loss = class_loss + 2.0 * seg_loss
        print(f"  Total loss: {total_loss.item():.4f}")

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("✗ Loss is NaN or Inf!")
            return False
        print("✓ Loss computation OK")
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dice_coefficient():
    """Test if dice_coefficient computation hangs or fails."""
    print("\nTesting dice_coefficient...")

    dummy_pred = torch.rand(2, 4, 224, 224).to(device)
    dummy_true = torch.rand(2, 4, 224, 224).to(device)

    try:
        dice = dice_coefficient(dummy_pred, dummy_true)
        print(f"  Dice: {dice:.6f}")
        if torch.isnan(torch.tensor(dice)):
            print("✗ Dice is NaN!")
            return False
        print("✓ Dice coefficient OK")
        return True
    except Exception as e:
        print(f"✗ Dice coefficient failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading(max_samples=100):
    """Test if data loading hangs or causes issues."""
    print("\nTesting data loading...")

    try:
        print("  Collecting dataset paths...")
        image_paths, labels = collect_data()
        print(f"  Total images available: {len(image_paths)}")

        if len(image_paths) == 0:
            print("✗ No images found!")
            return False

        test_paths = image_paths[:min(max_samples, len(image_paths))]
        test_labels = labels[:min(max_samples, len(labels))]

        print(f"  Testing with {len(test_paths)} images...")

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = SegmentationDataset(test_paths, test_labels, transform=tf)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        print(f"  DataLoader created. Iterating through batches...")

        for i, (images, masks, batch_labels) in enumerate(loader):
            print(f"    Batch {i+1}: images {images.shape}, masks {masks.shape}, labels shape {batch_labels.shape}")

            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"✗ Batch {i+1} contains NaN/Inf in images!")
                return False
            if torch.isnan(masks).any() or torch.isinf(masks).any():
                print(f"✗ Batch {i+1} contains NaN/Inf in masks!")
                return False

            if i >= 2:
                break

        print("✓ Data loading OK")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_loop():
    """Simulate a mini validation loop."""
    print("\nTesting validation loop...")

    try:
        print("  Collecting small dataset...")
        image_paths, labels = collect_data()

        if len(image_paths) < 20:
            print("✗ Not enough images for validation test!")
            return False

        test_paths = image_paths[:20]
        test_labels = labels[:20]

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = SegmentationDataset(test_paths, test_labels, transform=tf)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        model = MultiTaskUNet(in_channels=3, num_classes=4).to(device)
        model.eval()

        class_crit = nn.CrossEntropyLoss()
        seg_crit = DiceBCELoss()

        val_loss = 0
        val_correct = val_total = 0
        val_dice_sum = 0.0

        print("  Running validation loop...")
        with torch.no_grad():
            for batch_idx, (inputs, target_masks, target_labels) in enumerate(loader):
                print(f"    Batch {batch_idx+1}...", end=" ", flush=True)

                inputs = inputs.to(device)
                target_masks = target_masks.to(device)
                target_labels = target_labels.to(device)

                cls_logits, pred_masks = model(inputs)

                loss = class_crit(cls_logits, target_labels) + 2.0 * seg_crit(pred_masks, target_masks)
                val_loss += loss.item()
                _, pred = cls_logits.max(1)
                val_total += target_labels.size(0)
                val_correct += pred.eq(target_labels).sum().item()
                val_dice_sum += dice_coefficient(pred_masks, target_masks)

                print(f"loss={loss.item():.4f}")

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        val_dice = val_dice_sum / len(loader) if len(loader) > 0 else 0

        print(f"  Validation complete: Acc={val_acc:.2f}%, Dice={val_dice:.4f}")
        print("✓ Validation loop OK")
        return True
    except Exception as e:
        print(f"✗ Validation loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"Device: {device}\n")
    print("=" * 60)

    results = []
    results.append(("Model forward pass", test_model_forward()))
    results.append(("Loss computation", test_loss_computation()))
    results.append(("Dice coefficient", test_dice_coefficient()))
    results.append(("Data loading", test_data_loading()))
    results.append(("Validation loop", test_validation_loop()))

    print("\n" + "=" * 60)
    print("SUMMARY:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
