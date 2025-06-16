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
    
def compute_avg_std(data, key):
    temp_values = {}
    counter = 0
    for fold_data in data.values():
        for ski_id in fold_data[key]:
            if ski_id not in temp_values:
                temp_values[ski_id] = np.zeros(len(fold_data[key][ski_id]))
            temp_values[ski_id] = [a + b for a, b in zip(fold_data[key][ski_id], temp_values[ski_id])]
        counter += 1

    for metric in temp_values.keys():
        temp_values[metric] = [total / counter for total in temp_values[metric]]

        # Second pass: Compute squared differences for std dev
    temp_squared_diff = {metric: np.zeros(len(temp_values[metric])) for metric in temp_values.keys()}
    
    for fold_data in data.values():
        for ski_id in fold_data[key]:
            squared_diffs = [(x - mean) ** 2 for x, mean in zip(fold_data[key][ski_id], temp_values[ski_id])]
            temp_squared_diff[ski_id] = [a + b for a, b in zip(squared_diffs, temp_squared_diff[ski_id])]

    std_values = {metric: [np.sqrt(total / counter) for total in temp_squared_diff[metric]] for metric in temp_squared_diff.keys()}

    return temp_values, std_values

def plot_individual_skier(train_avg, train_std, val_avg, val_std, root_dir):
    # X-axis (time points or steps)
    first_key = next(iter(train_avg))  # Get the first key
    length_of_first_list = len(train_avg[first_key])
    x_values = np.arange(1, length_of_first_list+1)

    n_skiers = len(train_avg)
    cols = 4  # Number of columns in the grid
    rows = (n_skiers + cols - 1) // cols  # Calculate number of rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()  # Flatten to make it easy to loop through

    # Loop through each skier and plot on their respective subplot
    for i, skier in enumerate(train_avg.keys()):
        ax = axes[i]
        
        # Get Training and Validation data for the skier
        train_mean = np.array(train_avg[skier])
        train_std_dev = np.array(train_std[skier])
        val_mean = np.array(val_avg[skier])
        val_std_dev = np.array(val_std[skier])
        
        # Plot Training Data
        ax.plot(x_values, train_mean, label=f'Train', linestyle='--', alpha=0.7, color='#21638F')
        ax.fill_between(x_values, train_mean - train_std_dev, train_mean + train_std_dev, alpha=0.2, color='#21638F')
        
        # Plot Validation Data
        ax.plot(x_values, val_mean, label=f'Val', color='#000058')
        ax.fill_between(x_values, val_mean - val_std_dev, val_mean + val_std_dev, alpha=0.2, color='#000058')
        
        # Titles and labels
        ax.set_title(f"Skier {skier} Accuracy with Standard Deviation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(50, 100)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)

    # Hide empty subplots if the number of skiers doesn't fill the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plot_filename = os.path.join(root_dir, f"individual_skiers.png")
    plt.savefig(plot_filename)
    plt.close()

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
    root_dir = os.path.join(root_dir, "run_cv_" + start_time)
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
        if metric not in ["train_skiers_accs", "val_skiers_accs"]:
            metric_values = np.array([metrics_dict[fold][metric] for fold in metrics_dict])
            avg_metrics[metric] = np.mean(metric_values, axis=0)
            std_metrics[metric] = np.std(metric_values, axis=0)
        elif metric == "train_skiers_accs":
            # Compute statistics
            skier_avg_train_acc, skier_std_train_acc = compute_avg_std(metrics_dict, "train_skiers_accs")
            
        elif metric == "val_skiers_accs":
            skier_avg_val_acc, skier_std_val_acc = compute_avg_std(metrics_dict, "val_skiers_accs")

    plot_individual_skier(skier_avg_train_acc, skier_std_train_acc, skier_avg_val_acc, skier_std_val_acc, root_dir)

    epochs = range(1, len(avg_metrics[train_metrics[0]]) + 1)

    plot_handles = []  # Store figures to optionally display later

    # Plot each metric with train & validation together
    for train_metric, val_metric in zip(train_metrics, val_metrics):
        if train_metric not in ["train_skiers_accs", "val_skiers_accs"]:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot training metric
            ax.plot(epochs, avg_metrics[train_metric], linestyle='--', label=f'Avg {train_metric}', color='#21638F')
            ax.fill_between(epochs, avg_metrics[train_metric] - std_metrics[train_metric], 
                            avg_metrics[train_metric] + std_metrics[train_metric], color='#21638F', alpha=0.2)

            # Plot validation metric
            ax.plot(epochs, avg_metrics[val_metric], label=f'Avg {val_metric}', color='#000058')
            ax.fill_between(epochs, avg_metrics[val_metric] - std_metrics[val_metric], 
                            avg_metrics[val_metric] + std_metrics[val_metric], color='#000058', alpha=0.2)

            train_metric_name = train_metric.replace('_', ' ').title()
            val_metric_name = val_metric.replace('_', ' ').title()

            ax.set_xlabel("Epoch")
            ax.set_ylabel(train_metric_name)  # Use training metric name for y-label
            ax.set_title(f"Mean and Standard Deviation of {train_metric_name} & {val_metric_name} Across Folds")
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


def plot_training_final_metrics(results, root):
    metrics = {
        "train_losses": "Loss",
        "train_accs": "Accuracy",
        "train_precisions": "Precision",
        "train_recalls": "Recall",
        "train_f1s": "F1 Score"
    }

    colors = ["#93c8ee", "#4d90d3", "#21638f", "#004e98", "#102c9d", "#000058"] 

    for i, (key, title) in enumerate(metrics.items()):
        plt.figure(figsize=(6, 4))  # Create a new figure for each plot
        plt.plot(range(1, len(results[key]) + 1), results[key], color=colors[i])
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"Training {title}")
        plt.grid()

        # Save the plot
        save_path = os.path.join(root, f"{key}.png")
        plt.savefig(save_path)

    # Plot skier accuracy per epoch if available
    if "train_skier_accs" in results and results["train_skier_accs"]:
        plt.figure(figsize=(10, 6))

        # Extract skier IDs from the first epoch
        all_skiers = set()
        for epoch_data in results["train_skier_accs"]:
            all_skiers.update(epoch_data.keys())  # Collect all skier IDs

        skier_accuracies = {skier: [] for skier in all_skiers}

        # Populate accuracy values per skier for each epoch
        for epoch_data in results["train_skier_accs"]:
            for skier in skier_accuracies.keys():
                skier_accuracies[skier].append(epoch_data.get(skier, None))  # None if skier missing

        # Generate unique colors and line styles
        line_styles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]  # Different line styles

        # Plot each skier's accuracy
        for i, (skier_id, accuracy_list) in enumerate(skier_accuracies.items()):
            color_index = i % 6  # Cycle through 10 colors first
            style_index = i // 6  # Only change line style after one full color cycle
            
            color = colors[color_index]
            line_style = line_styles[style_index % len(line_styles)]  # Ensure we don't run out of styles

            plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, 
                     label=f"Skier {skier_id}", color=color, linestyle=line_style, linewidth=2)


        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Skier-wise Accuracy Over Epochs")
        #plt.ylim(0, 100)  # Accuracy is a percentage
        plt.legend(title="Skier ID", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid()

        # Save the plot
        save_path = os.path.join(root, "skier_accuracy.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    


def plot_lines(output_path, title, xlabel, ylabel, *line_data, labels=None, colors=None):
    
    """
    Plots multiple lines in a single figure. Used for showing DTW 

    Parameters:
    - output_path: The file path to save the plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - *line_data: Variable number of arrays to be plotted.
    - labels: Optional list of labels for each line.
    - colors: Optional list of colors for each line.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each line provided in *line_data
    for i, data in enumerate(line_data):
        label = labels[i] if labels else f'Line {i + 1}'
        color = colors[i] if colors else None
        plt.plot(data, linestyle='-', label=label, color=color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()