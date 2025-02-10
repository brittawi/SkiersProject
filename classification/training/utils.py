import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset
import json
import os
import matplotlib.pyplot as plt

# Utility functions

def training(train_loader, net, criterion, optimizer, device):
    running_loss = 0.0
    total_samples = 0  # To track the number of samples for accuracy calculation
    correct_predictions = 0  # To track the correct predictions
    all_labels = []
    all_predictions = []
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
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

def validation(val_loader, net, criterion, device):
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