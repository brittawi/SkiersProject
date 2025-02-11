import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Utility functions

def training(train_loader, net, criterion, optimizer, device, mlp):
    running_loss = 0.0
    total_samples = 0  # To track the number of samples for accuracy calculation
    correct_predictions = 0  # To track the correct predictions
    all_labels = []
    all_predictions = []
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if mlp:
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

def validation(val_loader, net, criterion, device, mlp):
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
            if mlp:
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

def train_and_validate(seed, net, criterion, optimizer, epochs, learning_rate, patience, train_loader, val_loader, device, mlp = False):
    set_seed(seed)

    results = {
        "train_losses": [],
        "train_accs": [],
        "train_precisions": [],
        "train_recalls": [],
        "train_f1s": [],
        "val_losses": [],
        "val_accs": [],
        "val_precisions": [],
        "val_recalls": [],
        "val_f1s": []
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):  # loop over the dataset multiple times
        
        print("Training")
        epoch_train_loss, epoch_train_acc, train_precision, train_recall, train_f1, train_conf_matrix = training(train_loader, net, criterion, optimizer, device, mlp)
        results["train_losses"].append(epoch_train_loss)
        results["train_accs"].append(epoch_train_acc)
        results["train_precisions"].append(train_precision)
        results["train_recalls"].append(train_recall)
        results["train_f1s"].append(train_f1)

        print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_train_loss:.3f}, Accuracy: {epoch_train_acc:.3f}")
        
        print("Validation")
        epoch_val_loss, epoch_val_acc, val_precision, val_recall, val_f1, val_conf_matrix = validation(val_loader, net, criterion, device, mlp)
        results["val_losses"].append(epoch_val_loss)
        results["val_accs"].append(epoch_val_acc)
        results["val_precisions"].append(val_precision)
        results["val_recalls"].append(val_recall)
        results["val_f1s"].append(val_f1)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_val_loss:.3f}, Accuracy: {epoch_val_acc:.3f}")

        # Check early stopping based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # TODO Check more later if loss or any other metric
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")  # Format as HH:MM:SS
            torch.save(net.state_dict(), f"best_model_{current_time}_lr{learning_rate}_seed{seed}.pth")  # Save the best model
            print(f"Model saved at epoch {epoch+1}")
            counter = 0  # Reset patience counter
            best_val_cm = val_conf_matrix
            best_train_cm = train_conf_matrix
        else:
            counter += 1  # Increment counter if no loss improvement
        
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break  # Stop training
        
    print('Finished Training')
    return results, best_train_cm, best_val_cm

def set_seed(seed=42):
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for CuDNN (ensures deterministic behavior)

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