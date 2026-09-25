"""
Lung-constrained attention loss for the DenseNet-121 classifier.

"Right for the right reasons" (Ross et al., 2017) applied to LungLens: penalise
the share of the class-activation map that falls outside the lung field, so the
classifier cannot lean on burned-in scanner text, the cardiac silhouette or the
image margin. 55% of the served overlays put most of their region outside the
lungs, which is the failure this targets.

The CAM here comes from the final DenseNet features and the linear head. For a
global-average-pool + linear head it is proportional to Grad-CAM at that layer,
so it is exact, costs nothing extra and needs no second-order gradients. The
served app still visualises Grad-CAM at denseblock4.denselayer16.conv2; the
evaluation script measures that map, not this one, so training and evaluation
do not share a shortcut.
"""
import torch
import torch.nn.functional as F

# Lung masks whose predicted area falls outside this band are treated as
# segmenter failures (the lung U-Net was trained on one source only), and the
# attention penalty is skipped for those images.
MIN_LUNG_AREA = 0.10
MAX_LUNG_AREA = 0.70
# Dilation keeps pleural, apical and hilar evidence from being penalised.
DEFAULT_DILATE_PX = 7
_EPS = 1e-6


def forward_with_features(cnn_model, x):
    """
    Run app.CNNModel and return (logits, rectified final features).

    Mirrors torchvision's DenseNet.forward exactly (features -> ReLU -> global
    average pool -> linear), so the logits are identical to cnn_model(x).
    """
    dense = cnn_model.cnnmodel
    feats = F.relu(dense.features(x))
    pooled = torch.flatten(F.adaptive_avg_pool2d(feats, (1, 1)), 1)
    return dense.classifier(pooled), feats


def class_activation_map(features, classifier_weight, class_idx):
    """
    CAM for one class per sample.

    features: (B, C, h, w), classifier_weight: (K, C), class_idx: (B,) long.
    Returns (B, h, w), rectified as Grad-CAM is.
    """
    w = classifier_weight[class_idx]
    cam = torch.einsum("bc,bchw->bhw", w, features)
    return F.relu(cam)


def outside_lung_fraction(cam, lung_mask, valid=None):
    """
    Mean fraction of activation energy outside the lung field.

    cam: (B, h, w) non-negative, any resolution; upsampled to the mask size.
    lung_mask: (B, H, W) in [0, 1].
    valid: optional (B,) bool; False rows are excluded.
    Returns a scalar in [0, 1]. Samples with no activation carry nothing to
    penalise and are excluded; if nothing is left, returns a zero that stays
    attached to the graph so backward() still works.
    """
    cam_up = F.interpolate(cam.unsqueeze(1), size=lung_mask.shape[-2:],
                           mode="bilinear", align_corners=False).squeeze(1)
    total = cam_up.sum(dim=(1, 2))
    outside = (cam_up * (1.0 - lung_mask)).sum(dim=(1, 2))
    keep = total > _EPS
    if valid is not None:
        keep = keep & valid.to(keep.device)
    if not keep.any():
        return cam.sum() * 0.0
    return (outside[keep] / total[keep]).mean()


@torch.no_grad()
def lung_masks_from_batch(lung_net, images, mean, std,
                          dilate_px=DEFAULT_DILATE_PX,
                          min_area=MIN_LUNG_AREA, max_area=MAX_LUNG_AREA):
    """
    Segment the lungs on an already-augmented, ImageNet-normalised batch.

    Running the segmenter after augmentation keeps flips, crops and rotations
    aligned with the image the classifier actually sees. The lung U-Net takes a
    single grey channel in [0, 1], matching fig7_work/lungfield.py.

    Returns (mask (B, H, W) float in {0, 1}, valid (B,) bool). Validity is
    judged on the undilated area.
    """
    mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    grey = (images * std_t + mean_t).clamp(0.0, 1.0).mean(dim=1, keepdim=True)
    mask = (torch.sigmoid(lung_net(grey)) > 0.5).float()
    area = mask.mean(dim=(1, 2, 3))
    valid = (area >= min_area) & (area <= max_area)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=dilate_px)
    return mask.squeeze(1), valid
