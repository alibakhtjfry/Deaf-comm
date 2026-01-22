import os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_metrics(model_dir):
    """
    Plot training loss and validation metrics from validation logs.
    
    Args:
        model_dir: Path to the model directory containing 'validations.txt'
    """
    validations_file = os.path.join(model_dir, "validations.txt")
    
    if not os.path.exists(validations_file):
        print(f"Error: Validation file not found at {validations_file}")
        return
    
    # Read validation data
    epochs = []
    steps = []
    losses = []
    val_dtw = []
    
    with open(validations_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Skip header
        try:
            parts = line.strip().split()
            if len(parts) >= 4:
                epochs.append(int(parts[0]))
                steps.append(int(parts[1]))
                losses.append(float(parts[2]))
                val_dtw.append(float(parts[3]))
        except (ValueError, IndexError):
            continue
    
    if not epochs:
        print("No validation data found")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot loss
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation DTW metric
    ax2.plot(epochs, val_dtw, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('DTW (Validation)')
    ax2.set_title('Validation DTW Metric')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(model_dir, "training_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Display plot
    plt.show()


def plot_tensorboard_logs(model_dir):
    """
    Plot metrics from TensorBoard logs (requires tensorboard package).
    
    Args:
        model_dir: Path to the model directory containing 'tensorboard' subdirectory
    """
    try:
        from torch.utils.tensorboard import summary
        import tensorflow as tf
    except ImportError:
        print("TensorBoard not available. Using validation file instead.")
        return False
    
    tb_dir = os.path.join(model_dir, "tensorboard")
    if not os.path.exists(tb_dir):
        print(f"TensorBoard directory not found at {tb_dir}")
        return False
    
    # Read events from TensorBoard logs
    losses = []
    val_metrics = []
    steps = []
    
    for event_file in os.listdir(tb_dir):
        if event_file.startswith('events.out.tfevents'):
            for event in tf.compat.v1.train.summary_iterator(os.path.join(tb_dir, event_file)):
                for value in event.summary.value:
                    if 'loss' in value.tag.lower():
                        steps.append(event.step)
                        losses.append(value.simple_value)
                    elif 'val' in value.tag.lower() or 'dtw' in value.tag.lower():
                        val_metrics.append(value.simple_value)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot Training Metrics")
    parser.add_argument("model_dir", type=str, 
                       help="Path to model directory")
    args = parser.parse_args()
    
    plot_training_metrics(args.model_dir)
