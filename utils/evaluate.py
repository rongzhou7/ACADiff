"""
Evaluation Metrics
Compute MAE, PSNR, SSIM, NMI for generated images
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import normalized_mutual_info_score
import os
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from inference import inference_pipeline


def compute_mae(pred, target, mask=None):
    """
    Mean Absolute Error
    pred, target: [N, 1, D, H, W] or [1, D, H, W]
    mask: optional brain mask [D, H, W]
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]

    mae = np.mean(np.abs(pred - target))
    return mae


def compute_psnr(pred, target, mask=None, data_range=2.0):
    """
    Peak Signal-to-Noise Ratio
    data_range: range of data (default 2.0 for [-1, 1])
    """
    pred = pred.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if mask is not None:
        pred = pred * mask
        target = target * mask

    # Compute PSNR for each sample
    if len(pred.shape) == 4:  # [N, D, H, W]
        psnr_values = []
        for i in range(pred.shape[0]):
            p = psnr(target[i], pred[i], data_range=data_range)
            psnr_values.append(p)
        return np.mean(psnr_values)
    else:  # [D, H, W]
        return psnr(target, pred, data_range=data_range)


def compute_ssim(pred, target, mask=None, data_range=2.0):
    """
    Structural Similarity Index
    """
    pred = pred.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if mask is not None:
        pred = pred * mask
        target = target * mask

    # Compute SSIM for each sample
    if len(pred.shape) == 4:  # [N, D, H, W]
        ssim_values = []
        for i in range(pred.shape[0]):
            s = ssim(target[i], pred[i], data_range=data_range)
            ssim_values.append(s)
        return np.mean(ssim_values)
    else:  # [D, H, W]
        return ssim(target, pred, data_range=data_range)


def compute_nmi(pred, target, mask=None, bins=256):
    """
    Normalized Mutual Information
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    if mask is not None:
        mask_flat = mask.flatten()
        pred = pred[mask_flat > 0]
        target = target[mask_flat > 0]

    # Discretize to bins
    pred_bins = np.digitize(pred, bins=np.linspace(pred.min(), pred.max(), bins))
    target_bins = np.digitize(target, bins=np.linspace(target.min(), target.max(), bins))

    nmi = normalized_mutual_info_score(target_bins, pred_bins)
    return nmi


def create_brain_mask(image, threshold=0.1):
    """
    Create simple brain mask by thresholding
    image: [1, D, H, W] or [D, H, W]
    Returns: [D, H, W] binary mask
    """
    if len(image.shape) == 4:
        image = image.squeeze(0)

    image_np = image.cpu().numpy()

    # Normalize to [0, 1]
    img_norm = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    # Threshold
    mask = (img_norm > threshold).astype(np.float32)

    return mask


def evaluate_generation(pred, target, use_mask=True):
    """
    Evaluate generation quality with all metrics

    Args:
        pred: [N, 1, D, H, W] predicted images
        target: [N, 1, D, H, W] ground truth images
        use_mask: whether to use brain mask

    Returns:
        dict of metrics
    """
    N = pred.shape[0]

    mae_list = []
    psnr_list = []
    ssim_list = []
    nmi_list = []

    for i in tqdm(range(N), desc='Computing metrics'):
        pred_i = pred[i]  # [1, D, H, W]
        target_i = target[i]  # [1, D, H, W]

        # Create brain mask from target
        if use_mask:
            mask = create_brain_mask(target_i)
        else:
            mask = None

        # Compute metrics
        mae = compute_mae(pred_i, target_i, mask)
        psnr_val = compute_psnr(pred_i, target_i, mask)
        ssim_val = compute_ssim(pred_i, target_i, mask)
        nmi = compute_nmi(pred_i, target_i, mask)

        mae_list.append(mae)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        nmi_list.append(nmi)

    results = {
        'MAE': {
            'mean': np.mean(mae_list),
            'std': np.std(mae_list),
            'values': mae_list
        },
        'PSNR': {
            'mean': np.mean(psnr_list),
            'std': np.std(psnr_list),
            'values': psnr_list
        },
        'SSIM': {
            'mean': np.mean(ssim_list),
            'std': np.std(ssim_list),
            'values': ssim_list
        },
        'NMI': {
            'mean': np.mean(nmi_list),
            'std': np.std(nmi_list),
            'values': nmi_list
        }
    }

    return results


def print_metrics(results):
    """
    Pretty print evaluation results
    """
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    for metric_name in ['MAE', 'PSNR', 'SSIM', 'NMI']:
        mean = results[metric_name]['mean']
        std = results[metric_name]['std']
        print(f"{metric_name:10s}: {mean:.4f} ± {std:.4f}")

    print("="*60 + "\n")


def evaluate_all_scenarios(split='test', num_samples=10, device='cuda'):
    """
    Evaluate all generation scenarios
    VBM from FDG+AV45, FDG from VBM+AV45, AV45 from VBM+FDG
    """
    scenarios = [
        ('VBM', ['FDG', 'AV45']),
        ('FDG', ['VBM', 'AV45']),
        ('AV45', ['VBM', 'FDG'])
    ]

    all_results = {}

    for target, available in scenarios:
        print(f"\n{'='*60}")
        print(f"Evaluating: Generate {target} from {available}")
        print(f"{'='*60}")

        # Run inference
        try:
            generated, ground_truth, subject_ids = inference_pipeline(
                target_modality=target,
                available_modalities=available,
                split=split,
                num_samples=num_samples,
                save_results=True,
                device=device
            )

            # Evaluate
            results = evaluate_generation(generated, ground_truth, use_mask=True)
            print_metrics(results)

            all_results[f'{target}_from_{"_".join(available)}'] = results

        except Exception as e:
            print(f"Error evaluating {target} from {available}: {e}")
            continue

    # Save all results
    save_dir = os.path.join(Config.SAVE_DIR, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)

    import json
    with open(os.path.join(save_dir, f'evaluation_{split}.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for scenario, res in all_results.items():
            results_serializable[scenario] = {
                metric: {
                    'mean': float(data['mean']),
                    'std': float(data['std'])
                }
                for metric, data in res.items()
            }
        json.dump(results_serializable, f, indent=2)

    print(f"\nAll results saved to {save_dir}")

    return all_results


def compare_with_baselines(split='test'):
    """
    Load and compare with baseline methods
    This is a placeholder - implement when you have baseline results
    """
    print("\n" + "="*60)
    print("Comparison with Baselines")
    print("="*60)

    # Load ACADiff results
    result_path = os.path.join(Config.SAVE_DIR, 'evaluation', f'evaluation_{split}.json')

    if not os.path.exists(result_path):
        print("Run evaluation first!")
        return

    import json
    with open(result_path, 'r') as f:
        acadiff_results = json.load(f)

    print("\nACADiff Results:")
    for scenario, metrics in acadiff_results.items():
        print(f"\n{scenario}:")
        for metric_name, values in metrics.items():
            print(f"  {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")

    # TODO: Load baseline results (Pix2Pix, DS-GAN, LDM, PASTA, FICD)
    # and compare

    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Generation Quality')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['single', 'all', 'compare'],
                        help='Evaluation mode')
    parser.add_argument('--target', type=str, default='VBM',
                        choices=['VBM', 'FDG', 'AV45'],
                        help='Target modality (for single mode)')
    parser.add_argument('--available', type=str, nargs='+', default=['FDG', 'AV45'],
                        help='Available modalities (for single mode)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of Monte Carlo samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--use_saved', action='store_true',
                        help='Use saved inference results instead of regenerating')

    args = parser.parse_args()

    if args.mode == 'single':
        # Evaluate single scenario
        if args.use_saved:
            # Load saved results
            result_path = os.path.join(
                Config.SAVE_DIR, 'inference',
                args.target, f'inference_{args.split}.pt'
            )
            data = torch.load(result_path)
            generated = data['generated']
            ground_truth = data['ground_truth']
        else:
            # Generate new
            generated, ground_truth, _ = inference_pipeline(
                target_modality=args.target,
                available_modalities=args.available,
                split=args.split,
                num_samples=args.num_samples,
                device=args.device
            )

        # Evaluate
        results = evaluate_generation(generated, ground_truth, use_mask=True)
        print_metrics(results)

    elif args.mode == 'all':
        # Evaluate all scenarios
        all_results = evaluate_all_scenarios(
            split=args.split,
            num_samples=args.num_samples,
            device=args.device
        )

    elif args.mode == 'compare':
        # Compare with baselines
        compare_with_baselines(split=args.split)
