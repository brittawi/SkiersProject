import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import json
import scipy.ndimage
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from sklearn.model_selection import KFold
import easydict
import yaml
import glob
from scipy.ndimage import gaussian_filter1d
from nets import LSTMNet, SimpleMLP

# ======================================
#      TRAIN AND VALIDATION SECTION
# ======================================

def update_config(config_file):
    with open(config_file) as f:
        config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        return config

METRICS_NAMES = ["train_losses",
        "train_accs",
        "train_precisions",
        "train_recalls",
        "train_f1s",
        "val_losses",
        "val_accs",
        "val_precisions",
        "val_recalls",
        "val_f1s"
    ]


def training(train_loader, net, criterion, optimizer, device, network_type):
    running_loss = 0.0
    total_samples = 0  # To track the number of samples for accuracy calculation
    correct_predictions = 0  # To track the correct predictions
    all_labels = []
    all_predictions = []
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        print(f"Current batch: {i+1}")
        inputs, labels = data
        if network_type == "mlp":
            inputs = inputs.view(inputs.size(0), -1)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()  # Add loss for this batch to the total loss
        total_samples += labels.size(0)  # Increment total samples
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        correct_predictions += (predicted == labels).sum().item()  # Increment correct predictions

        # For F1, recall, precision, confusion matrix
        all_labels.extend(labels.cpu().numpy())  # Store all labels
        all_predictions.extend(predicted.cpu().numpy())  # Store all predictions

    # Calculate the average loss for the epoch
    avg_epoch_loss = running_loss / len(train_loader)  # Average loss per batch
    epoch_accuracy = 100 * correct_predictions / total_samples  # Calculate accuracy as a percentage

    # Compute precision, recall, and F1-score
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return avg_epoch_loss, epoch_accuracy, precision, recall, f1, conf_matrix

def validation(val_loader, net, criterion, device, network_type):
    running_val_loss = 0.0
    epoch_accuracy = 0.0
    total_samples = 0  # To track the number of samples for accuracy calculation
    correct_predictions = 0  # To track the correct predictions
    all_labels = []
    all_predictions = []
    
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if network_type == "mlp":
                inputs = inputs.view(inputs.size(0), -1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            outputs = outputs.squeeze(1)
            val_loss = criterion(outputs, labels)
            
            # accumulate validation loss
            running_val_loss += val_loss.item()
            total_samples += labels.size(0)  # Increment total samples
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            correct_predictions += (predicted == labels).sum().item()  # Increment correct predictions

            # For F1, recall, precision, confusion matrix
            all_labels.extend(labels.cpu().numpy())  # Store all labels
            all_predictions.extend(predicted.cpu().numpy())  # Store all predictions
            
        # Calculate the average validation loss over all batches
        avg_val_loss = running_val_loss / len(val_loader)  # len(val_loader) is the number of batches
        epoch_accuracy = 100 * correct_predictions / total_samples  # Calculate accuracy as a percentage

        # Compute precision, recall, and F1-score
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return avg_val_loss, epoch_accuracy, precision, recall, f1, conf_matrix


def plot_single_metric(epoch_range, train_metric, val_metric, metric_name, xlabel, ylabel):
    plt.plot(epoch_range, train_metric, label=f'Training {metric_name}')
    plt.plot(epoch_range, val_metric, label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def train_and_validate(seed, net, criterion, optimizer, cfg, train_loader, val_loader, device):
    set_seed(seed)

    results = {}
    for metric_name in METRICS_NAMES:
        results[metric_name] = []

    best_val_acc = 0.0
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(cfg.TRAIN.EPOCHS):  # loop over the dataset multiple times
        
        print("Training")
        epoch_train_loss, epoch_train_acc, train_precision, train_recall, train_f1, train_conf_matrix = training(train_loader, net, criterion, optimizer, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["train_losses"].append(epoch_train_loss)
        results["train_accs"].append(epoch_train_acc)
        results["train_precisions"].append(train_precision)
        results["train_recalls"].append(train_recall)
        results["train_f1s"].append(train_f1)

        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_train_loss:.3f}, Accuracy: {epoch_train_acc:.3f}")
        
        print("Validation")
        epoch_val_loss, epoch_val_acc, val_precision, val_recall, val_f1, val_conf_matrix = validation(val_loader, net, criterion, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["val_losses"].append(epoch_val_loss)
        results["val_accs"].append(epoch_val_acc)
        results["val_precisions"].append(val_precision)
        results["val_recalls"].append(val_recall)
        results["val_f1s"].append(val_f1)
        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_val_loss:.3f}, Accuracy: {epoch_val_acc:.3f}")

        # Check early stopping based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # TODO We should use loss (Asked Homam)
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")  # Format as HH:MM:SS
            torch.save(net.state_dict(), f"best_model_{current_time}_lr{cfg.TRAIN.LR}_seed{seed}.pth")  # Save the best model
            print(f"Model saved at epoch {epoch+1}")
            counter = 0  # Reset patience counter
            best_val_cm = val_conf_matrix
            best_train_cm = train_conf_matrix
        else:
            counter += 1  # Increment counter if no loss improvement
        
        if counter >= cfg.TRAIN.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break  # Stop training
        
    print('Finished Training')
    return results, best_train_cm, best_val_cm

def cross_validation(cfg, fold_loaders, output_channels, device):
    all_results = []
    best_train_cms = []
    best_val_cms = []   
    
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n>>> Training on Fold {fold+1} <<<\n")
        if cfg.TRAIN.NETWORK.NETWORKTYPE == "lstm":
            input_channels = train_loader.dataset.data.shape[1]
        elif cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
            input_channels = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]
        
        # Initialize a new model for each fold
        for seed in cfg.TRAIN.SEEDS:
            print(f"\n========== Running for Seed {seed} on Fold {fold+1} ==========\n")
            
            # Set seed for reproducibility
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Create neural network
            net = initialize_net(cfg, input_channels, output_channels)
            net.to(device)

            # Define loss function and optimizer
            criterion_type = cfg.TRAIN.get('LOSS', "cross_entropy")
            if criterion_type == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss()
            else:
                print("Loss type not implemented")
            
            optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAIN.LR)  

            # Train and validate
            results, best_train_cm, best_val_cm = train_and_validate(seed, 
                                                                    net, 
                                                                    criterion, 
                                                                    optimizer,
                                                                    cfg,
                                                                    train_loader,
                                                                    val_loader,
                                                                    device,
                                                                    )
            
            # Store results
            all_results.append({'seed': seed, 'fold': fold+1, 'results': results})
            best_train_cms.append({'seed': seed, 'fold': fold+1, 'cm': best_train_cm})
            best_val_cms.append({'seed': seed, 'fold': fold+1, 'cm': best_val_cm})
    return all_results, best_train_cms, best_val_cms

def set_seed(seed=42):
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for CuDNN (ensures deterministic behavior)

def initialize_net(cfg, input_channels, output):
    net_type = cfg.TRAIN.get('NET', "mlp")
    
    if net_type == "lstm":
        print("Initializing lstm...")
        net = LSTMNet(input_channels, 
                    cfg.TRAIN.NETWORK.LSTM.HIDDEN_SIZE, 
                    output, 
                    cfg.TRAIN.NETWORK.LSTM.NUM_LAYERS, 
                    cfg.TRAIN.NETWORK.LSTM.DROPOUT)
    else:
        print("Initializing mlp...")
        net = SimpleMLP(input_channels, 
                        cfg.TRAIN.NETWORK.MLP.HIDDEN_1, 
                        cfg.TRAIN.NETWORK.MLP.HIDDEN_2, 
                        output)
        
    return net

#ChatGPT generated plotting

# Function to compute mean and std across seeds
def compute_mean_std(results, metric):
    metric_values = np.array([res[metric] for res in results])  # Shape: (num_seeds, num_epochs)
    mean_values = np.mean(metric_values, axis=0)  # Mean across seeds
    std_values = np.std(metric_values, axis=0)    # Std deviation across seeds
    return mean_values, std_values

# Plot function for train & val together
def plot_train_val(results, metric, title, ylabel, mean_std_results):
    epochs = len(results[0]["train_losses"])
    x = np.arange(1, epochs + 1)

    train_mean, train_std = mean_std_results[metric]["train"]
    val_mean, val_std = mean_std_results[metric]["val"]

    plt.figure(figsize=(8, 5))

    # Plot train
    plt.plot(x, train_mean, label="Train", color="blue")
    plt.fill_between(x, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)

    # Plot validation
    plt.plot(x, val_mean, label="Validation", color="red")
    plt.fill_between(x, val_mean - val_std, val_mean + val_std, color="red", alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot a metric
def plot_metric(metric, title, x, ylabel, mean_std_results):
    mean_values, std_values = mean_std_results[metric]

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_values, label=f"Mean {metric}", color="blue")
    plt.fill_between(x, mean_values - std_values, mean_values + std_values, color="blue", alpha=0.2, label="Std Dev")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# half ChatPGT generated
class CustomDatasetOld(Dataset):
    def __init__(self, 
                 path, 
                 choosen_joints=["RAnkle_x"],
                 label_dict = {
                    "unknown": 0,
                    "gear2" : 1,
                    "gear3" : 2,
                    "gear4" : 3,},
                 transform=None,
                 target_transform=None, 
                 padding_value=0.0,
                 apply_gaussian_filter=True,
                 mean=None, 
                 std=None,
                 ):
        
        self.label_dict = label_dict
        self.choosen_joints = choosen_joints
        self.padding_value = padding_value
        self.data, self.labels = self.__load_data(path)
        self.transform = transform
        self.target_transform = target_transform
        self.apply_gaussian_filter = apply_gaussian_filter
        self.mean = mean
        self.std = std

    def __load_data(self, path):
        data = []
        labels = []
        longest_cycle = 0  # Track longest cycle length

        for file in glob.glob(path + '/*.json'):
            with open(file, 'r') as f:
                data_json = json.load(f)

            for cycle in data_json.values():
                # Extract joint data as (num_joints, time_steps)
                cycle_data = [torch.tensor(cycle[joint], dtype=torch.float32) for joint in self.choosen_joints]

                # Stack into a (num_joints, time_steps) tensor
                cycle_tensor = torch.stack(cycle_data)  # Shape: (num_joints, time_steps)
                longest_cycle = max(longest_cycle, cycle_tensor.shape[1])  # Update max length

                data.append(cycle_tensor)
                labels.append(cycle["Label"])

        # Pad all cycles to match the longest cycle length
        padded_data = []
        for cycle in data:
            num_joints, time_steps = cycle.shape

            # Pad the time_steps dimension
            pad_length = longest_cycle - time_steps
            padded_cycle = torch.nn.functional.pad(cycle, (0, pad_length), value=self.padding_value)  # Pad last dim

            padded_data.append(padded_cycle)
        # Stack all padded cycles into a final tensor
        padded_data = torch.stack(padded_data)  # Shape: (num_cycles, num_joints, max_time)
        return padded_data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        label = self.label_dict[self.labels[idx]]
        item = self.data[idx].T #TODO Make sure this transpose makes sense for mlp

        if self.apply_gaussian_filter:
            item = torch.tensor(scipy.ndimage.gaussian_filter1d(item.numpy(), sigma=2, axis=0), dtype=torch.float32)
        
        if self.mean is not None and self.std is not None:
            item = (item - self.mean) / self.std
        
        if self.transform:
            item = self.transform(item)

        if self.target_transform:
            label = self.target_transform(label)

        return item, label

# half ChatPGT generated
class CustomDataset(Dataset):
    def __init__(self, 
                 data,
                 labels, 
                 label_dict = {
                    "unknown": 0,
                    "gear2" : 1,
                    "gear3" : 2,
                    "gear4" : 3,}
                 ):
        
        self.data = data
        self.labels = labels
        self.label_dict = label_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        label = self.label_dict[self.labels[idx]]
        item = torch.tensor(self.data[idx], dtype=torch.float32)
        item = self.data[idx].T #TODO Make sure this transpose makes sense for mlp
        
        return item, label
    
def calc_avg_metrics(k_folds, all_results, seeds, epochs):
    fold_final_results = {}
    for i in range(1, k_folds+1):
        fold_results = [entry for entry in all_results if entry['fold'] == i]
        results_lists = {}
        for metric_name in METRICS_NAMES:
            results_lists[metric_name] = np.zeros(epochs)
        for results in fold_results:

            for metric, values in results["results"].items():
                results_lists[metric] = [a + b for a, b in zip(results_lists[metric], values)]

        for metric in results_lists.keys():
            results_lists[metric] = [total / len(seeds) for total in results_lists[metric]]
            
        fold_final_results[i] = results_lists
    return fold_final_results
    
# TODO Option for fixing padding -> split load and split into train and train and validation splits first, then find longest in train and apply on validation
# Move loading out from CustomDataset to achieve 
def create_train_val_dataloaders(path, choosen_joints, train_size, val_size, k_folds, batch_size, seed = 42):
    # Create initial dataset (without normalization)
    train_dataset = CustomDataset(path, choosen_joints, padding_value=float('nan'), apply_gaussian_filter=False)

    # Split into Train+Val and Test
    generator = torch.Generator().manual_seed(seed)
    dataset_size = len(train_dataset)
    train_val_size = int(dataset_size * (train_size + val_size))  # 90% for Train+Val
    test_size = dataset_size - train_val_size  # 10% for Test

    # Get indices for Train+Val and Test
    #TODO test loader
    train_val_indices, test_indices = random_split(range(dataset_size), [train_val_size, test_size], generator=generator)

    # Initialize KFold (5 splits)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    train_val_indices = list(train_val_indices)  # Convert to list
    fold_loaders = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        print(f"Fold {fold+1}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
        
        # Get actual dataset indices
        train_subset_indices = [train_val_indices[i] for i in train_idx]
        val_subset_indices = [train_val_indices[i] for i in val_idx]

        # Create Subsets (before normalization)
        train_subset = Subset(train_dataset, train_subset_indices)

        # Compute mean and std from the training subset
        all_train_samples = torch.cat([train_subset[i][0] for i in range(len(train_subset))], dim=0)
        mean = all_train_samples.nanmean(dim=0)
        std = torch.std(all_train_samples[~torch.isnan(all_train_samples)])

        # Create a new dataset with computed mean/std (normalize train and val using train stats)
        normalized_dataset = CustomDataset(path, choosen_joints, mean=mean, std=std, apply_gaussian_filter=True)

        # Create train/val subsets on normalized dataset
        train_data = Subset(normalized_dataset, train_subset_indices)
        val_data = Subset(normalized_dataset, val_subset_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Store loaders for this fold
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders

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
                
# ======================================
#           PREPROCESSING SECTION
# ======================================

def normalize_per_timestamp(train_val_data, mean, std):
    """
    Normalizes data per timestamp and per joint.
    
    Parameters:
    - train_val_data: list of NumPy arrays of shape (num_joints, time_steps)

    Returns:
    - Normalized list of NumPy arrays with the same shape
    """
    
    # Normalize (avoid division by zero with epsilon)
    normalized_data = (train_val_data - mean) / (std + 1e-8)
    
    return normalized_data
    
def normalize_full_signal(train_val_data, mean, std):
    """
    Normalizes data per joint over the full signal (all timestamps concatenated).
    
    Parameters:
    - train_val_data: list of NumPy arrays of shape (num_joints, time_steps)

    Returns:
    - Normalized list of NumPy arrays with the same shape
    """

    # # Reshape to merge time steps: (num_joints, num_samples * time_steps)
    flattened = train_val_data.transpose(1, 0, 2).reshape(train_val_data.shape[1], -1)

    # Normalize each joint across all time steps
    normalized_flattened = (flattened - mean) / (std + 1e-8)

    # Reshape back to (num_samples, num_joints, time_steps)
    normalized_data = normalized_flattened.reshape(train_val_data.shape[1], train_val_data.shape[0], train_val_data.shape[2])
    normalized_data = normalized_data.transpose(1, 0, 2)  # Back to (num_samples, num_joints, time_steps)
    
    return normalized_data
    
def pad_sequences(sequences, max_length=None, pad_value=0.0):
    """
    Pads each sequence in the list to the specified max_length with a custom value.
    
    Parameters:
    - sequences: list of NumPy arrays (each of shape (num_joints, time_steps))
    - max_length: the desired length of the time series dimension (default None, will use the max length of sequences in sequences)
    - pad_value: the value to use for padding (default is 0.0)

    Returns:
    - List of padded NumPy arrays with shape (num_joints, max_length)
    """
    # Determine max_length if not provided
    if max_length is None:
        max_length = max(seq.shape[1] for seq in sequences)
    
    padded_sequences = []
    
    for seq in sequences:
        pad_length = max_length - seq.shape[1]
        
        # Pad if needed
        if pad_length > 0:
            padded_seq = np.pad(seq, ((0, 0), (0, pad_length)), mode='constant', constant_values=pad_value)
        # cut sequence if needed
        elif pad_length < 0:
            padded_seq = seq[:, :max_length]
        else:
            padded_seq = seq  # If already max_length, keep as is
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

def replace_nan_with_first_value(arr):
    """
    Replaces NaN values in each sequence with the first non-NaN value.
    
    Parameters:
        arr (numpy.ndarray): Input array of shape (samples, joints, time_steps).
    
    Returns:
        numpy.ndarray: Array with NaNs replaced.
    """
    mask = np.isnan(arr)  # Find NaN positions
    for sample in range(arr.shape[0]):  # Iterate over samples
        for joint in range(arr.shape[1]):  # Iterate over joints
            # Find the first non-NaN value
            valid_values = arr[sample, joint, ~mask[sample, joint]]
            if valid_values.size > 0:  # If there are valid values
                # TODO 
                first_value = valid_values[-1]  # Take the first valid number
                arr[sample, joint, mask[sample, joint]] = first_value  # Replace NaNs with it

    return arr
               
def preprocess_data(cfg, X_train, X_val, y_train, fold, plotting=False):
    # Pad the sequences to have the same length in both X_train and X_val
    max_length = max(seq.shape[1] for seq in X_train)  # Find the max length in X_train
    X_train = pad_sequences(X_train, max_length=max_length, pad_value=float('nan'))
    X_val = pad_sequences(X_val, max_length=max_length, pad_value=float('nan'))
    
    # TODO just for sanity checking
    X_train_raw = X_train
        
    # Normalize the padded training data
    if cfg.DATASET.AUG.NORMALIZATION:
        print("Normalizing the data...")
        
        norm_type = cfg.DATASET.AUG.get('NORM_TYPE', "full_signal")
        if norm_type == "per_timestamp":
            print("Normalizing the signal per timestamp...")
            
            # compute mean and std per timestamp
            mean = np.nanmean(X_train, axis=0)  # Shape: (num_joints, time_steps)
            std = np.nanstd(X_train, axis=0) 
            
            # normalize the train and val data
            X_train = normalize_per_timestamp(X_train, mean, std)
            X_val = normalize_per_timestamp(X_val, mean, std)
        
        else:
            print("Normalizing the full signal...")
            
            # Reshape to merge time steps: (num_joints, num_samples * time_steps)
            X_train_flattened = X_train.transpose(1, 0, 2).reshape(X_train.shape[1], -1)

            # Compute mean and std along the flattened axis
            mean = np.nanmean(X_train_flattened, axis=1, keepdims=True)  # Shape: (num_joints, 1)
            std = np.nanstd(X_train_flattened, axis=1, keepdims=True)    # Shape: (num_joints, 1)
            
            # normalize the train and val data
            X_train = normalize_full_signal(X_train, mean, std)
            X_val = normalize_full_signal(X_val, mean, std)
    
    # TODO when to replace padding?!!
    # replace padded values with start of sequence
    X_train = replace_nan_with_first_value(X_train)
    X_val = replace_nan_with_first_value(X_val)
    
    # Smooth the signal
    if cfg.DATASET.AUG.SMOOTHING > 0:
        print("Smoothing the signal")
        X_train = gaussian_filter1d(X_train, sigma=cfg.DATASET.AUG.SMOOTHING, axis=2)
        
        
    if plotting and fold == 0:
        plot_raw_vs_normalized(X_train_raw, X_train, y_train)
        
    return X_train, X_val, mean, std, max_length