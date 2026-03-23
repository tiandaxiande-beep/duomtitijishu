import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ColorHistogramCBIR:
	"""CBIR system based on global color histograms."""

	def __init__(self, bins: Tuple[int, int, int] = (8, 8, 8)) -> None:
		self.bins = bins
		self.index: List[Dict[str, object]] = []

	@staticmethod
	def _extract_label(file_name: str) -> str:
		"""Extract category name from leading alphabetic prefix in filename."""
		stem = os.path.splitext(os.path.basename(file_name))[0].lower()
		match = re.match(r"([a-zA-Z]+)", stem)
		return match.group(1).lower() if match else "unknown"

	def extract_histogram(self, image_bgr: np.ndarray) -> np.ndarray:
		"""Compute and normalize RGB 3D histogram as feature vector."""
		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		hist = cv2.calcHist(
			[image_rgb],
			[0, 1, 2],
			None,
			list(self.bins),
			[0, 256, 0, 256, 0, 256],
		)
		hist = hist.flatten().astype(np.float32)
		hist_sum = np.sum(hist)
		if hist_sum > 0:
			hist /= hist_sum
		return hist

	@staticmethod
	def similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
		"""Histogram intersection similarity in [0, 1]."""
		return float(np.minimum(hist_a, hist_b).sum())

	def build_index(self, dataset_dir: str) -> None:
		"""Build feature database from all images in dataset_dir."""
		self.index.clear()
		valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

		for file_name in sorted(os.listdir(dataset_dir)):
			path = os.path.join(dataset_dir, file_name)
			if not os.path.isfile(path):
				continue
			if os.path.splitext(file_name)[1].lower() not in valid_ext:
				continue

			image = cv2.imread(path)
			if image is None:
				continue

			self.index.append(
				{
					"path": path,
					"file": file_name,
					"label": self._extract_label(file_name),
					"hist": self.extract_histogram(image),
				}
			)

		if not self.index:
			raise ValueError(f"No valid images found in dataset: {dataset_dir}")

	def search(self, query_image_path: str, top_k: int = 3) -> List[Dict[str, object]]:
		"""Retrieve top_k most similar images for query image."""
		if not self.index:
			raise ValueError("Dataset index is empty. Call build_index() first.")

		query_img = cv2.imread(query_image_path)
		if query_img is None:
			raise FileNotFoundError(f"Cannot read query image: {query_image_path}")

		query_hist = self.extract_histogram(query_img)
		results: List[Dict[str, object]] = []
		for item in self.index:
			score = self.similarity(query_hist, item["hist"])  # type: ignore[index]
			results.append(
				{
					"path": item["path"],
					"file": item["file"],
					"label": item["label"],
					"score": score,
				}
			)

		results.sort(key=lambda x: x["score"], reverse=True)
		return results[:top_k]

	def rank_all(self, query_image_path: str) -> List[Dict[str, object]]:
		"""Return full ranked list for evaluation."""
		if not self.index:
			raise ValueError("Dataset index is empty. Call build_index() first.")

		query_img = cv2.imread(query_image_path)
		if query_img is None:
			raise FileNotFoundError(f"Cannot read query image: {query_image_path}")

		query_hist = self.extract_histogram(query_img)
		ranked: List[Dict[str, object]] = []
		for item in self.index:
			score = self.similarity(query_hist, item["hist"])  # type: ignore[index]
			ranked.append(
				{
					"path": item["path"],
					"file": item["file"],
					"label": item["label"],
					"score": score,
				}
			)

		ranked.sort(key=lambda x: x["score"], reverse=True)
		return ranked


class CBIREvaluator:
	"""Evaluation for Precision, Recall and Average Precision (AP/MAP for one query)."""

	@staticmethod
	def precision_recall(
		ranked_results: List[Dict[str, object]],
		query_label: str,
		total_relevant: int,
	) -> Tuple[List[float], List[float]]:
		tp = 0
		precision_list: List[float] = []
		recall_list: List[float] = []

		for rank, item in enumerate(ranked_results, start=1):
			if item["label"] == query_label:
				tp += 1
			precision_list.append(tp / rank)
			recall_list.append(tp / total_relevant if total_relevant > 0 else 0.0)

		return precision_list, recall_list

	@staticmethod
	def average_precision(
		ranked_results: List[Dict[str, object]], query_label: str, total_relevant: int
	) -> float:
		if total_relevant <= 0:
			return 0.0

		tp = 0
		precision_at_hits: List[float] = []
		for rank, item in enumerate(ranked_results, start=1):
			if item["label"] == query_label:
				tp += 1
				precision_at_hits.append(tp / rank)

		return float(np.mean(precision_at_hits)) if precision_at_hits else 0.0

	def evaluate(
		self,
		ranked_results: List[Dict[str, object]],
		query_label: str,
		total_relevant: int,
	) -> Dict[str, object]:
		precision_list, recall_list = self.precision_recall(
			ranked_results, query_label, total_relevant
		)
		ap = self.average_precision(ranked_results, query_label, total_relevant)

		final_precision = precision_list[-1] if precision_list else 0.0
		final_recall = recall_list[-1] if recall_list else 0.0

		return {
			"precision_curve": precision_list,
			"recall_curve": recall_list,
			"AP": ap,
			"MAP": ap,
			"final_precision": final_precision,
			"final_recall": final_recall,
		}


def _resolve_query_path(project_root: str, preferred: Optional[str] = None) -> str:
	"""Resolve query path; supports typo-style query.jps as fallback."""
	candidates = []
	if preferred:
		candidates.append(preferred)
	candidates.extend(["query.jps", "query.jpg", "query.jpeg", "query.png", "query.bmp"])

	for name in candidates:
		path = name if os.path.isabs(name) else os.path.join(project_root, name)
		if os.path.isfile(path):
			return path

	raise FileNotFoundError(
		"Query image not found. Expected one of: query.jps/query.jpg/query.jpeg/query.png/query.bmp"
	)


def _plot_query_and_pr(
	query_image_path: str,
	recall_curve: List[float],
	precision_curve: List[float],
	map_score: float,
) -> None:
	query_img_bgr = cv2.imread(query_image_path)
	if query_img_bgr is None:
		return
	query_img_rgb = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2RGB)

	plt.figure(figsize=(10, 4))
	ax1 = plt.subplot(1, 2, 1)
	ax1.plot(recall_curve, precision_curve, color="blue")
	ax1.set_title(f"Precision-Recall Curve | MAP: {map_score:.4f}")
	ax1.set_xlabel("Recall")
	ax1.set_ylabel("Precision")
	ax1.set_xlim(0, 1.0)
	ax1.set_ylim(0, 1.05)
	ax1.grid(True, linestyle="--", alpha=0.5)

	ax2 = plt.subplot(1, 2, 2)
	ax2.imshow(query_img_rgb)
	ax2.set_title("Query Image")
	ax2.axis("off")

	plt.tight_layout()


def _plot_top_results(results: List[Dict[str, object]]) -> None:
	if not results:
		return

	n = len(results)
	plt.figure(figsize=(4 * n, 4))
	for i, item in enumerate(results, start=1):
		img_bgr = cv2.imread(str(item["path"]))
		if img_bgr is None:
			continue
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

		ax = plt.subplot(1, n, i)
		ax.imshow(img_rgb)
		ax.set_title(f"Top{i}\n{item['file']}\nscore: {item['score']:.4f}")
		ax.axis("off")

	plt.tight_layout()


def run_experiment(
	dataset_dir: str = "dataset",
	query_path: Optional[str] = None,
	top_k: int = 3,
	bins: Tuple[int, int, int] = (8, 8, 8),
) -> Dict[str, object]:
	"""Run complete CBIR workflow: indexing, retrieval, evaluation, visualization."""
	project_root = os.path.dirname(os.path.abspath(__file__))
	dataset_path = dataset_dir if os.path.isabs(dataset_dir) else os.path.join(project_root, dataset_dir)
	query_image_path = _resolve_query_path(project_root, query_path)

	cbir = ColorHistogramCBIR(bins=bins)
	cbir.build_index(dataset_path)

	ranked_results = cbir.rank_all(query_image_path)
	top_results = ranked_results[:top_k]

	query_label = ColorHistogramCBIR._extract_label(os.path.basename(query_image_path))
	all_labels = [str(item["label"]) for item in cbir.index]
	label_set = set(all_labels)

	# If query label is unknown or not present in dataset classes,
	# fall back to top-1 retrieved label for single-query evaluation.
	if (query_label == "unknown" or query_label not in label_set) and top_results:
		query_label = str(top_results[0]["label"])

	total_relevant = sum(1 for label in all_labels if label == query_label)
	evaluator = CBIREvaluator()
	metrics = evaluator.evaluate(ranked_results, query_label, total_relevant)

	print("=" * 60)
	print("Color Histogram CBIR Experiment")
	print(f"Dataset dir     : {dataset_path}")
	print(f"Query image     : {query_image_path}")
	print(f"Query label     : {query_label}")
	print(f"Histogram bins  : {bins}")
	print(f"Total images    : {len(cbir.index)}")
	print(f"Total relevant  : {total_relevant}")
	print("-" * 60)
	print("Top-K Retrieval Results")
	for idx, item in enumerate(top_results, start=1):
		print(
			f"Top{idx}: {item['file']} | label={item['label']} | similarity={item['score']:.4f}"
		)
	print("-" * 60)
	print(f"Precision: {metrics['final_precision']:.4f}")
	print(f"Recall   : {metrics['final_recall']:.4f}")
	print(f"MAP(AP)  : {metrics['MAP']:.4f}")
	print("=" * 60)

	_plot_query_and_pr(
		query_image_path,
		metrics["recall_curve"],  # type: ignore[arg-type]
		metrics["precision_curve"],  # type: ignore[arg-type]
		float(metrics["MAP"]),
	)
	_plot_top_results(top_results)
	plt.show()

	return {
		"top_results": top_results,
		"metrics": metrics,
		"query_path": query_image_path,
		"query_label": query_label,
	}


if __name__ == "__main__":
	run_experiment(dataset_dir="dataset", query_path=None, top_k=3, bins=(8, 8, 8))
