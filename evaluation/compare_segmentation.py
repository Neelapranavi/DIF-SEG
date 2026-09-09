import json
from pathlib import Path


def main(
    baseline_path="evaluation_results/segmentation_metrics.json",
    augmented_path="evaluation_results/segmentation_diffusion_metrics.json",
):
    baseline_file = Path(baseline_path)
    augmented_file = Path(augmented_path)
    if not baseline_file.exists() or not augmented_file.exists():
        raise FileNotFoundError(
            "Run baseline and diffusion-augmented segmentation evaluation first."
        )

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    augmented = json.loads(augmented_file.read_text(encoding="utf-8"))
    comparison = {
        "baseline": baseline,
        "diffusion_inspired": augmented,
        "delta_mean_dice": float(augmented["mean_dice"] - baseline["mean_dice"]),
        "delta_mean_iou": float(augmented["mean_iou"] - baseline["mean_iou"]),
    }

    output = Path("evaluation_results/segmentation_comparison.json")
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
