"""Restore one RGB image or a folder of images with regression or I2I-RFR.

Default: I2I-RFR with three Euler steps.
Use --standard_regression for a checkpoint trained with that flag.
Both input and result are saved as NAME_in.png and NAME_out.png.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from baseline_network import Network

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def infer_image(model, source, destination, device, n_steps=3, standard_regression=False):
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        x = T.ToTensor()(rgb).unsqueeze(0).to(device)  # [0, 1]

    with torch.inference_mode():
        if standard_regression:
            y = model(x)
        else:  # I2I-RFR: explicit Euler integration backward in time
            y_t = torch.randn_like(x)
            for step in range(n_steps):
                t = 1.0 - step / n_steps
                velocity = (y_t - model(torch.cat((x, y_t), dim=1))) / t
                y_t = y_t - velocity / n_steps
            y = y_t

    output_image = T.ToPILImage()(y[0].clamp(0, 1).cpu())
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_path = destination.with_name(f"{destination.stem}_in.png")
    output_path = destination.with_name(f"{destination.stem}_out.png")
    rgb.save(input_path)
    output_image.save(output_path)
    return input_path, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_sampling_steps", type=int, default=3)
    parser.add_argument("--standard_regression", action="store_true")
    args = parser.parse_args()
    
    if args.n_sampling_steps < 1:
        parser.error("--n_sampling_steps must be >= 1")    

    source = Path(args.input)
    destination = Path(args.output)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image: {source}")
        target = (destination if destination.suffix.lower() in IMAGE_EXTENSIONS
                  else destination / f"{source.stem}.png")
        image_pairs = [(source, target)]
    elif source.is_dir():
        if destination.suffix.lower() in IMAGE_EXTENSIONS:
            raise ValueError("For folder input, --output must be a folder")
        files = sorted(p for p in source.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        image_pairs = [(p, destination / p.relative_to(source).with_suffix(".png"))
                       for p in files]
    else:
        raise FileNotFoundError(f"Input not found: {source}")
    if not image_pairs:
        raise FileNotFoundError(f"No images found in {source}")

    torch.manual_seed(10)  # Reproducible Gaussian initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3 if args.standard_regression else 6
    model = Network(in_channels=in_channels, out_channels=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ("standard_regression" in checkpoint and
            checkpoint["standard_regression"] != args.standard_regression):
        raise ValueError("Checkpoint regression mode does not match --standard_regression")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    for i, (src, dst) in enumerate(image_pairs, 1):
        input_path, output_path = infer_image(
            model, src, dst, device, args.n_sampling_steps, args.standard_regression)
        print(f"[{i}/{len(image_pairs)}] {input_path}, {output_path}", flush=True)


if __name__ == "__main__":
    main()
