from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_rgb332_palette() -> np.ndarray:
	"""Build a 256-color RGB332 palette using bit replication for display."""
	palette = np.zeros((256, 3), dtype=np.uint8)
	for idx in range(256):
		r3 = (idx >> 5) & 0x07
		g3 = (idx >> 2) & 0x07
		b2 = idx & 0x03

		# Bit replication maps low-bit channels to full 8-bit range.
		r = (r3 << 5) | (r3 << 2) | (r3 >> 1)
		g = (g3 << 5) | (g3 << 2) | (g3 >> 1)
		b = (b2 << 6) | (b2 << 4) | (b2 << 2) | b2
		palette[idx] = np.array([r, g, b], dtype=np.uint8)
	return palette


def quantize_rgb332(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Quantize RGB image to RGB332 indices and reconstructed RGB image."""
	r3 = rgb[:, :, 0] >> 5
	g3 = rgb[:, :, 1] >> 5
	b2 = rgb[:, :, 2] >> 6

	indices = ((r3 << 5) | (g3 << 2) | b2).astype(np.uint8)
	palette = build_rgb332_palette()
	reconstructed = palette[indices]
	return indices, reconstructed


def mse_psnr(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
	diff = original.astype(np.float32) - reconstructed.astype(np.float32)
	mse = float(np.mean(diff * diff))
	if mse == 0:
		return mse, float("inf")
	psnr = 10.0 * np.log10((255.0**2) / mse)
	return mse, float(psnr)


def save_indexed_png(indices: np.ndarray, palette: np.ndarray, output_path: Path) -> None:
	img_p = Image.fromarray(indices, mode="P")
	img_p.putpalette(palette.reshape(-1).tolist())
	img_p.save(output_path)


def main() -> None:
	parser = argparse.ArgumentParser(description="RGB332 8-bit color quantization")
	parser.add_argument("--input", type=Path, default=Path("test.png.png"), help="input image path")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("rgb332_quantized.png"),
		help="output indexed PNG path",
	)
	parser.add_argument(
		"--recon",
		type=Path,
		default=Path("rgb332_reconstructed.png"),
		help="output reconstructed RGB path",
	)
	args = parser.parse_args()

	rgb = np.array(Image.open(args.input).convert("RGB"), dtype=np.uint8)
	indices, reconstructed = quantize_rgb332(rgb)
	palette = build_rgb332_palette()

	save_indexed_png(indices, palette, args.output)
	Image.fromarray(reconstructed, mode="RGB").save(args.recon)

	mse, psnr = mse_psnr(rgb, reconstructed)
	print(f"Input: {args.input}")
	print(f"Indexed PNG: {args.output}")
	print(f"Reconstructed RGB: {args.recon}")
	print(f"MSE: {mse:.4f}")
	print(f"PSNR: {psnr:.4f} dB")


if __name__ == "__main__":
	main()
