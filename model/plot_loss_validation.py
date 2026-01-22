import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class TrainingMetricsPlotter:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.validations_file = os.path.join(model_dir, "validations.txt")

    def parse_validations_file(self) -> Tuple[List[int], List[int], List[float], List[float]]:
        epochs, steps, losses, val_dtw = [], [], [], []

        if not os.path.exists(self.validations_file):
            print(f"Warning: validations.txt not found at {self.validations_file}")
            return epochs, steps, losses, val_dtw

        with open(self.validations_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse format: "Steps: X Loss: Y| DTW: Z| LR: ..."
                    if 'Steps:' in line and 'Loss:' in line and 'DTW:' in line:
                        # Extract Steps
                        step_part = line.split('Steps:')[1].split('Loss:')[0].strip()
                        step = int(step_part)
                        
                        # Extract Loss
                        loss_part = line.split('Loss:')[1].split('|')[0].strip()
                        loss = float(loss_part)
                        
                        # Extract DTW
                        dtw_part = line.split('DTW:')[1].split('|')[0].strip()
                        dtw = float(dtw_part)
                        
                        epochs.append(idx)  # Use line index as epoch proxy
                        steps.append(step)
                        losses.append(loss)
                        val_dtw.append(dtw)
                except (ValueError, IndexError):
                    # skip any malformed line
                    continue

        return epochs, steps, losses, val_dtw

    @staticmethod
    def _safe_norm(values: List[float]) -> np.ndarray:
        x = np.array(values, dtype=float)
        denom = x.max() - x.min()
        return (x - x.min()) / denom if denom != 0 else np.zeros_like(x)

    def plot_loss_and_validation(self, save_path: str = None):
        epochs, steps, losses, val_dtw = self.parse_validations_file()

        if not epochs:
            print("No data to plot (validations.txt parsed but empty).")
            return

        fig = plt.figure(figsize=(14, 10))

        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(epochs, losses, linewidth=2, marker="o", markersize=5, label="Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss vs Epoch")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(epochs, val_dtw, linewidth=2, marker="s", markersize=5, label="DTW")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("DTW")
        ax2.set_title("Validation DTW vs Epoch")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(steps, losses, linewidth=1.5, marker=".", markersize=3)
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss vs Steps")
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 2, 4)
        loss_n = self._safe_norm(losses)
        dtw_n = self._safe_norm(val_dtw)
        ax4.plot(epochs, loss_n, linewidth=2, marker="o", markersize=5, label="Loss (norm)")
        ax4.plot(epochs, dtw_n, linewidth=2, marker="s", markersize=5, label="DTW (norm)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Normalized")
        ax4.set_title("Normalized Loss & DTW")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim([0, 1])

        plt.suptitle("Training Metrics Summary", fontsize=14, y=0.995)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.model_dir, "metrics_plot.png")

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

        self._save_statistics(epochs, losses, val_dtw)
        plt.show()

    def _save_statistics(self, epochs: List[int], losses: List[float], val_dtw: List[float]):
        stats_file = os.path.join(self.model_dir, "metrics_statistics.txt")

        min_loss_i = int(np.argmin(losses))
        max_loss_i = int(np.argmax(losses))
        min_dtw_i = int(np.argmin(val_dtw))
        max_dtw_i = int(np.argmax(val_dtw))

        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("Training Metrics Statistics\n")
            f.write("=" * 50 + "\n\n")

            f.write("Loss Statistics:\n")
            f.write(f"  Minimum Loss: {losses[min_loss_i]:.6f} (Epoch {epochs[min_loss_i]})\n")
            f.write(f"  Maximum Loss: {losses[max_loss_i]:.6f} (Epoch {epochs[max_loss_i]})\n")
            f.write(f"  Average Loss: {np.mean(losses):.6f}\n")
            f.write(f"  Std Dev Loss: {np.std(losses):.6f}\n\n")

            f.write("DTW Validation Statistics:\n")
            f.write(f"  Minimum DTW: {val_dtw[min_dtw_i]:.6f} (Epoch {epochs[min_dtw_i]})\n")
            f.write(f"  Maximum DTW: {val_dtw[max_dtw_i]:.6f} (Epoch {epochs[max_dtw_i]})\n")
            f.write(f"  Average DTW: {np.mean(val_dtw):.6f}\n")
            f.write(f"  Std Dev DTW: {np.std(val_dtw):.6f}\n\n")

            f.write("Training Summary:\n")
            f.write(f"  Total Epoch Records: {len(epochs)}\n")
            f.write(f"  Loss Change: {(losses[-1] - losses[0]):.6f} (end - start)\n")
            f.write(f"  DTW Change: {(val_dtw[-1] - val_dtw[0]):.6f} (end - start)\n")

        print(f"Statistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser("Plot training metrics (loss + validation DTW)")
    parser.add_argument("model_dir", type=str, help="Path to model directory containing validations.txt")
    parser.add_argument("--save", type=str, default=None, help="Save plot path (default: model_dir/metrics_plot.png)")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found")
        return

    TrainingMetricsPlotter(args.model_dir).plot_loss_and_validation(save_path=args.save)


if __name__ == "__main__":
    main()
