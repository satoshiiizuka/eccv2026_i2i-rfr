"""Train paired image-to-image translation with I2I-RFR or standard L1 regression.

Default: I2I-RFR (noisy GT and t-reweighted L1 loss).
Use --standard_regression for ordinary one-pass L1 regression.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_network import Network
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="datasets/train/low_blur_noise")
    parser.add_argument("--gt_root", default="datasets/train/high_sharp_scaled")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=20_000)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--standard_regression", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    dataset = utils.PairedDataset(args.input_root, args.gt_root, args.crop_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)

    in_channels = 3 if args.standard_regression else 6
    model = Network(in_channels=in_channels, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if ("standard_regression" in checkpoint and
                checkpoint["standard_regression"] != args.standard_regression):
            raise ValueError("Checkpoint regression mode does not match --standard_regression")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"]
        if scheduler.T_max != args.steps:
            raise ValueError("--steps must match the original training schedule")        
        print(f"Resumed from step {start_step:,}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mode = "Standard L1" if args.standard_regression else "I2I-RFR"
    print(f"Mode: {mode} | Pairs: {len(dataset.paths):,} | Device: {device}")

    model.train()
    data_iter = iter(loader)
    for step in range(start_step + 1, args.steps + 1):
        try:
            input, gt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input, gt = next(data_iter)

        input = input.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if args.standard_regression:
            loss = F.l1_loss(model(input), gt)
        else:  # I2I-RFR: Beta(2,1) time sampling and t-reweighted L1
            t = torch.rand(gt.size(0), 1, 1, 1, device=device).sqrt().clamp_min(1e-3)
            noise = torch.randn_like(gt)
            y_t = (1 - t) * gt + t * noise
            loss = ((model(torch.cat((input, y_t), dim=1)) - gt).abs() / t).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_interval == 0:
            print(f"step {step:,}/{args.steps:,} | {mode} loss {loss.item():.5f} "
                  f"| lr {scheduler.get_last_lr()[0]:.3e}", flush=True)

        if step % args.save_interval == 0 or step == args.steps:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "standard_regression": args.standard_regression,
            }, save_dir / "latest.pth")
            print(f"Saved at step {step:,}", flush=True)


if __name__ == "__main__":
    main()
