import matplotlib.pyplot as plt
import os
import numpy as np

def plot_raw_vs_normalized(X_train_raw, X_train, y_train):
    # Extract the raw signal (before normalization)
    raw_signal = X_train_raw[0].T  # From the first dataset instance
    label = y_train[0]

    # Extract the processed signal (after normalization)
    normalized_signal = X_train[0].T  # From the first dataset instance

    # Plot the signals for each joint
    plt.figure(figsize=(12, 6))

    for i in range(raw_signal.shape[1]):  # Iterate over each joint
        plt.subplot(1, 2, 1)
        plt.plot(raw_signal[:, i], label=f'Joint {i}')
        plt.title(f"Raw Signal (Before Normalization), label = {label}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(normalized_signal[:, i], label=f'Joint {i}')
        plt.title("Normalized Signal (After Normalization)")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_avg_std_combined(metrics_dict, cfg, start_time, show_plots=False):
    """
    Plots the average and standard deviation of training and validation metrics over folds,
    saves them as PNG files, and optionally displays them.

    Parameters:
    - metrics_dict (dict): Dictionary containing metrics for each fold.
    - cfg: Configuration object containing paths.
    - start_time (str): Used for creating save folder structure.
    - show_plots (bool, optional): If True, displays the plots after saving. Defaults to False.
    """

    # Ensure save directory exists
    root_dir = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.PLOT_PATH)
    root_dir = os.path.join(root_dir, "run_" + start_time)
    os.makedirs(root_dir, exist_ok=True)

    # Identify training and validation metric names dynamically
    first_fold = next(iter(metrics_dict))
    metric_keys = list(metrics_dict[first_fold].keys())

    # Separate train and validation metrics
    train_metrics = [m for m in metric_keys if "train" in m]
    val_metrics = [m.replace("train", "val") for m in train_metrics]  # Assume val metric names are similar

    # Initialize storage for averages and standard deviations
    avg_metrics = {}
    std_metrics = {}

    # Compute mean and std across folds for each metric
    for metric in metric_keys:
        metric_values = np.array([metrics_dict[fold][metric] for fold in metrics_dict])
        avg_metrics[metric] = np.mean(metric_values, axis=0)
        std_metrics[metric] = np.std(metric_values, axis=0)

    epochs = range(1, len(avg_metrics[train_metrics[0]]) + 1)

    plot_handles = []  # Store figures to optionally display later

    # Plot each metric with train & validation together
    for train_metric, val_metric in zip(train_metrics, val_metrics):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot training metric
        ax.plot(epochs, avg_metrics[train_metric], label=f'Avg {train_metric}', color='b')
        ax.fill_between(epochs, avg_metrics[train_metric] - std_metrics[train_metric], 
                         avg_metrics[train_metric] + std_metrics[train_metric], color='b', alpha=0.2)

        # Plot validation metric
        ax.plot(epochs, avg_metrics[val_metric], label=f'Avg {val_metric}', color='r')
        ax.fill_between(epochs, avg_metrics[val_metric] - std_metrics[val_metric], 
                         avg_metrics[val_metric] + std_metrics[val_metric], color='r', alpha=0.2)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(train_metric.replace('_', ' ').title())  # Use training metric name for y-label
        ax.set_title(f"Mean and Std of {train_metric} & {val_metric} Across Folds")
        ax.legend()
        ax.grid(True)

        # Save the plot as a PNG file
        plot_filename = os.path.join(root_dir, f"{train_metric}_vs_{val_metric}.png")
        plt.savefig(plot_filename)

        if show_plots:
            plot_handles.append(fig)
        else:
            plt.close(fig)  # Close the plot to prevent it from showing up

    # If show_plots is True, display all plots at once
    if show_plots:
        plt.show()


# # Plot function for train & val together
# def plot_train_val(results, metric, title, ylabel, mean_std_results):
#     epochs = len(results[0]["train_losses"])
#     x = np.arange(1, epochs + 1)

#     train_mean, train_std = mean_std_results[metric]["train"]
#     val_mean, val_std = mean_std_results[metric]["val"]

#     plt.figure(figsize=(8, 5))

#     # Plot train
#     plt.plot(x, train_mean, label="Train", color="blue")
#     plt.fill_between(x, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)

#     # Plot validation
#     plt.plot(x, val_mean, label="Validation", color="red")
#     plt.fill_between(x, val_mean - val_std, val_mean + val_std, color="red", alpha=0.2)

#     plt.xlabel("Epochs")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     plt.show()

# def plot_single_metric(epoch_range, train_metric, val_metric, metric_name, xlabel, ylabel):
#     plt.plot(epoch_range, train_metric, label=f'Training {metric_name}')
#     plt.plot(epoch_range, val_metric, label=f'Validation {metric_name}')
#     plt.title(f'Training and Validation {metric_name}')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()

# # Function to plot a metric
# def plot_metric(metric, title, x, ylabel, mean_std_results):
#     mean_values, std_values = mean_std_results[metric]

#     plt.figure(figsize=(8, 5))
#     plt.plot(x, mean_values, label=f"Mean {metric}", color="blue")
#     plt.fill_between(x, mean_values - std_values, mean_values + std_values, color="blue", alpha=0.2, label="Std Dev")
#     plt.xlabel("Epochs")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     plt.show()