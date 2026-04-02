from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def mse_psnr(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
	diff = original.astype(np.float32) - reconstructed.astype(np.float32)
	mse = float(np.mean(diff * diff))
	if mse == 0:
		return mse, float("inf")
	psnr = 10.0 * np.log10((255.0**2) / mse)
	return mse, float(psnr)


def split_box(unique_colors: np.ndarray, counts: np.ndarray, box_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
	"""Split one color box by largest-range channel at weighted median."""
	box_colors = unique_colors[box_indices]
	ranges = box_colors.max(axis=0) - box_colors.min(axis=0)
	channel = int(np.argmax(ranges))

	if ranges[channel] == 0:
		return None

	order = np.argsort(box_colors[:, channel], kind="mergesort")
	sorted_indices = box_indices[order]
	sorted_counts = counts[sorted_indices]

	cumulative = np.cumsum(sorted_counts)
	half = cumulative[-1] / 2.0
	split_pos = int(np.searchsorted(cumulative, half))
	split_pos = max(1, min(split_pos, len(sorted_indices) - 1))

	return sorted_indices[:split_pos], sorted_indices[split_pos:]


def median_cut_quantize(rgb: np.ndarray, palette_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return indexed image, palette, and reconstructed RGB image."""
	h, w, _ = rgb.shape
	flat = rgb.reshape(-1, 3)

	unique_colors, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)

	boxes: list[np.ndarray] = [np.arange(len(unique_colors), dtype=np.int32)]

	while len(boxes) < palette_size:
		split_candidate = -1
		best_range = -1

		for i, idxs in enumerate(boxes):
			if len(idxs) < 2:
				continue
			colors = unique_colors[idxs]
			box_range = int(np.max(colors.max(axis=0) - colors.min(axis=0)))
			if box_range > best_range:
				best_range = box_range
				split_candidate = i

		if split_candidate == -1:
			break

		current = boxes.pop(split_candidate)
		result = split_box(unique_colors, counts, current)
		if result is None:
			boxes.append(current)
			break
		left, right = result
		boxes.append(left)
		boxes.append(right)

	palette = np.zeros((len(boxes), 3), dtype=np.uint8)
	color_to_palette = np.zeros(len(unique_colors), dtype=np.uint8)

	for palette_idx, idxs in enumerate(boxes):
		weights = counts[idxs].astype(np.float64)
		weighted_sum = (unique_colors[idxs].astype(np.float64) * weights[:, None]).sum(axis=0)
		color = np.round(weighted_sum / weights.sum()).astype(np.uint8)
		palette[palette_idx] = color
		color_to_palette[idxs] = np.uint8(palette_idx)

	indices = color_to_palette[inverse].reshape(h, w)
	reconstructed = palette[indices]
	return indices, palette, reconstructed


def save_indexed_png(indices: np.ndarray, palette: np.ndarray, output_path: Path) -> None:
	# Indexed PNG palette must have exactly 256*3 entries.
	full_palette = np.zeros((256, 3), dtype=np.uint8)
	full_palette[: len(palette)] = palette

	img_p = Image.fromarray(indices, mode="P")
	img_p.putpalette(full_palette.reshape(-1).tolist())
	img_p.save(output_path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Median Cut 8-bit color quantization")
	parser.add_argument("--input", type=Path, default=Path("test.png.png"), help="input image path")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("median_cut_quantized.png"),
		help="output indexed PNG path",
	)
	parser.add_argument(
		"--recon",
		type=Path,
		default=Path("median_cut_reconstructed.png"),
		help="output reconstructed RGB path",
	)
	parser.add_argument("--colors", type=int, default=256, help="palette size (1-256)")
	args = parser.parse_args()

	if not 1 <= args.colors <= 256:
		raise ValueError("--colors must be in [1, 256]")

	rgb = np.array(Image.open(args.input).convert("RGB"), dtype=np.uint8)
	indices, palette, reconstructed = median_cut_quantize(rgb, palette_size=args.colors)

	save_indexed_png(indices, palette, args.output)
	Image.fromarray(reconstructed, mode="RGB").save(args.recon)

	mse, psnr = mse_psnr(rgb, reconstructed)
	print(f"Input: {args.input}")
	print(f"Indexed PNG: {args.output}")
	print(f"Reconstructed RGB: {args.recon}")
	print(f"Palette size used: {len(palette)}")
	print(f"MSE: {mse:.4f}")
	print(f"PSNR: {psnr:.4f} dB")


if __name__ == "__main__":
	main()
