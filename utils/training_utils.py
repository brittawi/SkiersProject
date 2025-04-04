import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import random
from utils.nets import LSTMNet, SimpleMLP
from utils.focal_loss import FocalLoss 


METRICS_NAMES = ["train_losses",
        "train_accs",
        "train_precisions",
        "train_recalls",
        "train_f1s",
        "train_skiers_accs",
        "val_losses",
        "val_accs",
        "val_precisions",
        "val_recalls",
        "val_f1s",
        "val_skiers_accs"
    ]

# ======================================
#      TRAIN AND VALIDATION SECTION
# ======================================


def training(train_loader, net, criterion, optimizer, device, network_type):
    """
    Trains the model for one epoch.

    Parameters:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - net (torch.nn.Module): Neural network model.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - network_type (str): Type of network ('mlp' or 'lstm').

    Returns:
    - avg_epoch_loss (float): Average training loss for the epoch.
    - epoch_accuracy (float): Training accuracy in percentage.
    - precision (float): Weighted precision score.
    - recall (float): Weighted recall score.
    - f1 (float): Weighted F1-score.
    - conf_matrix (numpy.ndarray): Confusion matrix of predictions.
    """

    running_loss = 0.0
    total_samples = 0  # To track the number of samples for accuracy calculation
    correct_predictions = 0  # To track the correct predictions
    all_labels = []
    all_predictions = []
    skier_accuracy = {}  # Dictionary to store accuracy per skier
    
    # setting model to training => bug fix (https://discuss.pytorch.org/t/cudnn-rnn-backward-can-only-be-called-in-training-mode/37622)
    net.train()
    
    for i, data in enumerate(train_loader, 0):
        if len(data) == 3:
            inputs, labels, skiers = data  # Case with skier IDs
            track_skiers = True
        else:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            skiers = None
            track_skiers = False
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

        # If skiers are provided, track accuracy per skier
        if track_skiers:
            skiers = skiers.cpu().numpy()
            labels_np = labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            for skier, true_label, pred_label in zip(skiers, labels_np, predicted_np):
                if skier not in skier_accuracy:
                    skier_accuracy[skier] = {"correct": 0, "total": 0}
                
                skier_accuracy[skier]["total"] += 1
                if true_label == pred_label:
                    skier_accuracy[skier]["correct"] += 1

    # Calculate the average loss for the epoch
    avg_epoch_loss = running_loss / len(train_loader)  # Average loss per batch
    epoch_accuracy = 100 * correct_predictions / total_samples  # Calculate accuracy as a percentage

    # Compute precision, recall, and F1-score
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Compute accuracy per skier
    if track_skiers:
        for skier in skier_accuracy:
            skier_accuracy[skier] = 100 * skier_accuracy[skier]["correct"] / skier_accuracy[skier]["total"]

    return avg_epoch_loss, epoch_accuracy, precision, recall, f1, conf_matrix, skier_accuracy if track_skiers else None

def validation(val_loader, net, criterion, device, network_type):
    """
    Evaluates the model on the validation set.

    Parameters:
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - net (torch.nn.Module): Trained neural network model.
    - criterion (torch.nn.Module): Loss function.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - network_type (str): Type of network ('mlp' or 'lstm').

    Returns:
    - avg_val_loss (float): Average validation loss.
    - epoch_accuracy (float): Validation accuracy in percentage.
    - precision (float): Weighted precision score.
    - recall (float): Weighted recall score.
    - f1 (float): Weighted F1-score.
    - conf_matrix (numpy.ndarray): Confusion matrix of predictions.
    """

    running_val_loss = 0.0
    epoch_accuracy = 0.0
    total_samples = 0  # To track the number of samples for accuracy calculation
    correct_predictions = 0  # To track the correct predictions
    all_labels = []
    all_predictions = []
    skier_accuracy = {}  # Dictionary to store accuracy per skier
    
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            if len(data) == 3:
                inputs, labels, skiers = data  # Case with skier IDs
                track_skiers = True
            else:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                skiers = None
                track_skiers = False
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

                            # If skiers are provided, track accuracy per skier
            if track_skiers:
                skiers = skiers.cpu().numpy()
                labels_np = labels.cpu().numpy()
                predicted_np = predicted.cpu().numpy()

                for skier, true_label, pred_label in zip(skiers, labels_np, predicted_np):
                    if skier not in skier_accuracy:
                        skier_accuracy[skier] = {"correct": 0, "total": 0}
                    
                    skier_accuracy[skier]["total"] += 1
                    if true_label == pred_label:
                        skier_accuracy[skier]["correct"] += 1

            
        # Calculate the average validation loss over all batches
        avg_val_loss = running_val_loss / len(val_loader)  # len(val_loader) is the number of batches
        epoch_accuracy = 100 * correct_predictions / total_samples  # Calculate accuracy as a percentage

        # Compute precision, recall, and F1-score
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Compute accuracy per skier
        if track_skiers:
            for skier in skier_accuracy:
                skier_accuracy[skier] = 100 * skier_accuracy[skier]["correct"] / skier_accuracy[skier]["total"]
    
    return avg_val_loss, epoch_accuracy, precision, recall, f1, conf_matrix, skier_accuracy if track_skiers else None

def train_and_validate(seed, net, criterion, optimizer, cfg, train_loader, val_loader, device, fold, start_time, custom_params):
    """
    Trains and validates the model across multiple epochs. Saves the model weights based on lowest validation accuracy. 

    Parameters:
    - seed (int): Random seed for reproducibility.
    - net (torch.nn.Module): Neural network model.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - cfg (Config): Configuration object containing training parameters.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - fold (int): Current cross validation fold. 
    - start_time (str): Start time as string for saving models.
    - custom_params (dict): Parameters that need to be saved with the state dict like mean of train set.

    Returns:
    - results (dict): Dictionary storing loss, accuracy, precision, recall, and F1-score for each epoch.
    - best_train_cm (numpy.ndarray): Best confusion matrix for training.
    - best_val_cm (numpy.ndarray): Best confusion matrix for validation.
    """

    # Set all seeds and make deterministic
    set_seed(seed)

    results = {}
    for metric_name in METRICS_NAMES:
        results[metric_name] = []

    best_val_acc = 0.0
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(cfg.TRAIN.EPOCHS):  # loop over the dataset multiple times
        
        print("Training")
        epoch_train_loss, epoch_train_acc, train_precision, train_recall, train_f1, train_conf_matrix, train_skier_acc = training(train_loader, net, criterion, optimizer, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["train_losses"].append(epoch_train_loss)
        results["train_accs"].append(epoch_train_acc)
        results["train_precisions"].append(train_precision)
        results["train_recalls"].append(train_recall)
        results["train_f1s"].append(train_f1)
        results["train_skiers_accs"].append(train_skier_acc)

        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_train_loss:.3f}, Accuracy: {epoch_train_acc:.3f}")
        print(f"Skier acc", train_skier_acc[1])
        
        print("Validation")
        epoch_val_loss, epoch_val_acc, val_precision, val_recall, val_f1, val_conf_matrix, val_skier_acc = validation(val_loader, net, criterion, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["val_losses"].append(epoch_val_loss)
        results["val_accs"].append(epoch_val_acc)
        results["val_precisions"].append(val_precision)
        results["val_recalls"].append(val_recall)
        results["val_f1s"].append(val_f1)
        results["val_skiers_accs"].append(val_skier_acc)
        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_val_loss:.3f}, Accuracy: {epoch_val_acc:.3f}")
        print(f"Skier acc val", train_skier_acc[1])

        # Check early stopping based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_model(net, cfg, fold, seed, start_time, custom_params)
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

def save_model(net, cfg, fold, seed, start_time, custom_params):
    """
    Saves the model weights in a structured folder based on fold and seed.

    Parameters:
    - net (torch.nn.Module): Neural network model.
    - cfg (Config): Configuration object containing training parameters.
    - fold (int): Current cross-validation fold.
    - seed (int): Random seed for reproducibility.
    - start_time (str): Time for timestamping model creation.
    - custom_params (dict): Parameters that need to be saved with the state dict like mean of train set.
    """

    # Define the folder structure
    run_dir = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.MODEL_DIR)
    model_dir = os.path.join(run_dir, f"run_{start_time}_{cfg.TRAIN.NETWORK.NETWORKTYPE}")
    fold_dir = os.path.join(model_dir, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    # Create a timestamped filename
    model_filename = f"best_model_{start_time}_lr{cfg.TRAIN.LR}_seed{seed}.pth"
    model_path = os.path.join(fold_dir, model_filename)

    # Save the model state dictionary
    #torch.save(net.state_dict(), model_path)
    torch.save({
        "state_dict": net.state_dict(),
        "custom_params" : custom_params
    }, model_path)

def cross_validation(cfg, fold_loaders, output_channels, device, start_time):
    """
    Performs k-fold cross-validation on the dataset. Creates a new net for each seed in each fold. 

    Parameters:
    - cfg (Config): Configuration object containing training settings.
    - fold_loaders (list): List of (train_loader, val_loader, custom_params) tuples for each fold and custom params per fold as a dict.
    - output_channels (int): Number of output classes.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - start_time (str): Time used for timestamping saved models/folder structure. 
    - custom_params (dict): Parameters that need to be saved with the state dict like mean of train set.

    Returns:
    - all_results (list): List of dictionaries storing results for each fold and seed.
    - best_train_cms (list): List of best training confusion matrices.
    - best_val_cms (list): List of best validation confusion matrices.
    """

    all_results = []
    best_train_cms = []
    best_val_cms = []
    
    for fold, (train_loader, val_loader, custom_params) in enumerate(fold_loaders):
        print(f"\n>>> Training on Fold {fold+1} <<<\n")
        if cfg.TRAIN.NETWORK.NETWORKTYPE == "lstm":
            input_channels = train_loader.dataset.data.shape[1]
        elif cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
            input_channels = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]
            
        custom_params["input_channels"] = input_channels
        custom_params["output_channels"] = output_channels
        
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
            criterion = initialize_loss(cfg)
            
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
                                                                    fold,
                                                                    start_time,
                                                                    custom_params
                                                                    )
            
            # Store results
            all_results.append({'seed': seed, 'fold': fold+1, 'results': results})
            best_train_cms.append({'seed': seed, 'fold': fold+1, 'cm': best_train_cm})
            best_val_cms.append({'seed': seed, 'fold': fold+1, 'cm': best_val_cm})
    return all_results, best_train_cms, best_val_cms

def set_seed(seed=42):
    """
    Sets seeds for reproducibility

    Parameters:
    - seed (int): Number to use as seed. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for CuDNN (ensures deterministic behavior)

def initialize_net(cfg, input_channels, output):
    net_type = cfg.TRAIN.NETWORK.get('NETWORKTYPE', "mlp")
    
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

def initialize_loss(cfg, config = None, num_classes=4):
    # Check if cfg is string, because can't send in cfg in hyperparameter optimization
    if not config == None:
        criterion_type = config["loss_type"]
    else:
        criterion_type = cfg.TRAIN.get('LOSS', "cross_entropy")
        num_classes = len(cfg.DATA_PRESET.LABELS.keys())

    if criterion_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type == "focal_loss":
        criterion = FocalLoss(num_classes=num_classes)
    else:
        print("Loss type not implemented")
    return criterion

def write_cv_results(fold_final_results, epochs, writer):
    # Loop through each fold
    for fold in fold_final_results:
        for epoch in range(epochs):  # Loop through epochs
            # Log train metrics
            writer.add_scalar(f'Fold_{fold}/Train/Loss', fold_final_results[fold]['train_losses'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Train/Accuracy', fold_final_results[fold]['train_accs'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Train/Precision', fold_final_results[fold]['train_precisions'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Train/Recall', fold_final_results[fold]['train_recalls'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Train/F1', fold_final_results[fold]['train_f1s'][epoch], epoch)
            
            # Log validation metrics
            writer.add_scalar(f'Fold_{fold}/Val/Loss', fold_final_results[fold]['val_losses'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Val/Accuracy', fold_final_results[fold]['val_accs'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Val/Precision', fold_final_results[fold]['val_precisions'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Val/Recall', fold_final_results[fold]['val_recalls'][epoch], epoch)
            writer.add_scalar(f'Fold_{fold}/Val/F1', fold_final_results[fold]['val_f1s'][epoch], epoch)
            
def calc_avg_metrics(k_folds, all_results, seeds, epochs):
    fold_final_results = {}

    for i in range(1, k_folds + 1):
        fold_results = [entry for entry in all_results if entry["fold"] == i]
        results_lists = {metric_name: np.zeros(epochs) for metric_name in METRICS_NAMES}

        # Collect all unique skier IDs across train and val
        all_train_skiers, all_val_skiers = set(), set()
        for results in fold_results:
            for epoch_data in results["results"].get("train_skiers_accs", []):
                if epoch_data:  # Ensure it's not None
                    all_train_skiers.update(epoch_data.keys())  # Collect all train skier IDs
            for epoch_data in results["results"].get("val_skiers_accs", []):
                if epoch_data:  # Ensure it's not None
                    all_val_skiers.update(epoch_data.keys())  # Collect all val skier IDs

        # Initialize skier accuracy storage
        train_skiers_accs = {skier: [0] * epochs for skier in all_train_skiers}
        val_skiers_accs = {skier: [0] * epochs for skier in all_val_skiers}

        # Sum metrics across seeds
        for results in fold_results:
            for metric, values in results["results"].items():
                if metric not in ["train_skiers_accs", "val_skiers_accs"]:
                    results_lists[metric] = [a + b for a, b in zip(results_lists[metric], values)]
                else:
                    # Sum skier accuracies per epoch
                    if metric == "train_skiers_accs":
                        for epoch_idx, epoch_data in enumerate(values):
                            if epoch_data:  # Ensure epoch_data is not None
                                for skier in all_train_skiers:
                                    train_skiers_accs[skier][epoch_idx] += epoch_data.get(skier, 0)

                    elif metric == "val_skiers_accs":
                        for epoch_idx, epoch_data in enumerate(values):
                            if epoch_data:  # Ensure epoch_data is not None
                                for skier in all_val_skiers:
                                    val_skiers_accs[skier][epoch_idx] += epoch_data.get(skier, 0)

        # Compute averages for metrics and skier accuracies
        for metric in results_lists.keys():
            results_lists[metric] = [total / len(seeds) for total in results_lists[metric]]

        for skier in train_skiers_accs.keys():
            train_skiers_accs[skier] = [total / len(seeds) for total in train_skiers_accs[skier]]

        for skier in val_skiers_accs.keys():
            val_skiers_accs[skier] = [total / len(seeds) for total in val_skiers_accs[skier]]

        results_lists["train_skiers_accs"] = train_skiers_accs
        results_lists["val_skiers_accs"] = val_skiers_accs
        fold_final_results[i] = results_lists

    return fold_final_results

def print_save_best_epoch(results, save_path, start_time, custom_params):
    # Extract the number of epochs from the first fold
    num_epochs = len(next(iter(results.values()))["val_accs"])

    # Initialize a dictionary to store metrics across folds
    avg_metrics = {metric: np.zeros(num_epochs) for metric in results[1].keys()}
    std_metrics = {metric: np.zeros(num_epochs) for metric in results[1].keys()}

    # Compute the average and standard deviation for each epoch
    for epoch in range(num_epochs):
        for metric in avg_metrics.keys():
            if metric not in ["train_skiers_accs", "val_skiers_accs"]:
                values = [fold_data[metric][epoch] for fold_data in results.values()]
                avg_metrics[metric][epoch] = np.mean(values)
                std_metrics[metric][epoch] = np.std(values)

    # Find the epoch with the highest average validation accuracy
    best_val_acc_epoch = np.argmax(avg_metrics["val_accs"])
    best_val_loss_epoch = np.argmin(avg_metrics["val_losses"])

    # Prepare the result string
    result_text = f"Epoch with highest val acc: {best_val_acc_epoch}\n"
    result_text += f"Best epoch with lowest val loss: {best_val_loss_epoch}\n"
    result_text += f"Folds: {len(values)}\n"
    result_text += f"Train Loss: {avg_metrics['train_losses'][best_val_loss_epoch]:.4f} ± {std_metrics['train_losses'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Train Acc: {avg_metrics['train_accs'][best_val_loss_epoch]:.2f}% ± {std_metrics['train_accs'][best_val_loss_epoch]:.2f}\n"
    result_text += f"Train Precision: {avg_metrics['train_precisions'][best_val_loss_epoch]:.4f} ± {std_metrics['train_precisions'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Train Recall: {avg_metrics['train_recalls'][best_val_loss_epoch]:.4f} ± {std_metrics['train_recalls'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Train F1 Score: {avg_metrics['train_f1s'][best_val_loss_epoch]:.4f} ± {std_metrics['train_f1s'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Val Loss: {avg_metrics['val_losses'][best_val_loss_epoch]:.4f} ± {std_metrics['val_losses'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Val Acc: {avg_metrics['val_accs'][best_val_loss_epoch]:.2f}% ± {std_metrics['val_accs'][best_val_loss_epoch]:.2f}\n"
    result_text += f"Val Precision: {avg_metrics['val_precisions'][best_val_loss_epoch]:.4f} ± {std_metrics['val_precisions'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Val Recall: {avg_metrics['val_recalls'][best_val_loss_epoch]:.4f} ± {std_metrics['val_recalls'][best_val_loss_epoch]:.4f}\n"
    result_text += f"Val F1 Score: {avg_metrics['val_f1s'][best_val_loss_epoch]:.4f} ± {std_metrics['val_f1s'][best_val_loss_epoch]:.4f}\n"

    # Print results to console
    print(result_text)

    for key, item in custom_params.items():
        result_text += str(key) + " " + str(item) + "\n"

    # Save to a text file
    output_file = f"best_epoch_{start_time}_results.txt"
    output_file = os.path.join(save_path, output_file)
    with open(output_file, "w") as file:
        file.write(result_text)

    print(f"Results saved to {output_file}")

