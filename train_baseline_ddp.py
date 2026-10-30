"""Train with I2I-RFR or standard L1 regression, on one or multiple GPUs.

Single GPU: python train_baseline_ddp.py
Multi GPU:  torchrun --standalone --nproc_per_node=2 train_baseline_ddp.py

--batch_size is PER GPU. Checkpoints are compatible with inference.py.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from baseline_network import Network
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="datasets/train/low_blur")
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

    # torchrun sets WORLD_SIZE and LOCAL_RANK. Plain python uses one process.
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    try:
        torch.manual_seed(10 + rank)
        torch.backends.cudnn.benchmark = (device.type == "cuda")

        dataset = utils.PairedDataset(args.input_root, args.gt_root, args.crop_size)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if distributed else None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(sampler is None),
            sampler=sampler, num_workers=args.workers,
            pin_memory=(device.type == "cuda"), drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError("Dataset too small for the requested GPUs and batch size")

        in_channels = 3 if args.standard_regression else 6
        network = Network(in_channels=in_channels, out_channels=3).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=1e-6)
        
        start_step = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            if ("standard_regression" in checkpoint and
                    checkpoint["standard_regression"] != args.standard_regression):
                raise ValueError("Checkpoint mode does not match --standard_regression")
            network.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])            
            start_step = checkpoint["step"]
            if scheduler.T_max != args.steps:
                raise ValueError("--steps must match the original training schedule")            

        model = DDP(network, device_ids=[local_rank] if device.type == "cuda" else None) if distributed else network
        mode = "Standard L1" if args.standard_regression else "I2I-RFR"
        if rank == 0:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            print(f"Mode: {mode} | Pairs: {len(dataset.paths):,} | GPUs/processes: {world_size} "
                  f"| Batch/device: {args.batch_size} | Global batch: {args.batch_size * world_size}", flush=True)
            if start_step:
                print(f"Resumed from step {start_step:,}", flush=True)

        model.train()
        epoch = (start_step + len(loader) - 1) // len(loader)  # Fresh shuffle on resume.
        if sampler is not None:
            sampler.set_epoch(epoch)
        data_iter = iter(loader)
        for step in range(start_step + 1, args.steps + 1):
            try:
                input, gt = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                input, gt = next(data_iter)

            input = input.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if args.standard_regression:
                loss = F.l1_loss(model(input), gt)
            else:  # I2I-RFR: Beta(2, 1) time sampling and t-reweighted L1
                t = torch.rand(gt.size(0), 1, 1, 1, device=device).sqrt().clamp_min(1e-3)
                noise = torch.randn_like(gt)
                y_t = (1 - t) * gt + t * noise
                loss = ((model(torch.cat((input, y_t), dim=1)) - gt).abs() / t).mean()

            loss.backward()  # DDP averages gradients across processes automatically.
            optimizer.step()
            scheduler.step()

            if step == 1 or step % args.log_interval == 0:
                mean_loss = loss.detach().clone()
                if distributed:
                    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                    mean_loss /= world_size
                if rank == 0:
                    print(f"step {step:,}/{args.steps:,} | {mode} loss {mean_loss.item():.5f} "
                          f"| lr {scheduler.get_last_lr()[0]:.3e}", flush=True)

            if (step % args.save_interval == 0 or step == args.steps) and rank == 0:
                torch.save({
                    "step": step,
                    "model": network.state_dict(),  # No DDP 'module.' prefix.
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "standard_regression": args.standard_regression,
                }, Path(args.save_dir) / "latest.pth")
                print(f"Saved at step {step:,}", flush=True)

        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
