import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics data
resnet_metrics = pd.read_csv('../exercises/1/metrics/resnet18_model_metrics_detailed.csv')
cnn_metrics = pd.read_csv('../exercises/1/metrics/cnn_model_metrics_detailed.csv')

# Function to plot Loss, Accuracy, Precision, Recall, F1-Score, and Training Time
def plot_metrics(resnet_metrics, cnn_metrics):
    epochs = resnet_metrics['Epoch']

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, resnet_metrics['Train Loss'], label='ResNet18 Train Loss', linestyle='--', marker='o')
    plt.plot(epochs, resnet_metrics['Validation Loss'], label='ResNet18 Val Loss', linestyle='-', marker='o')
    plt.plot(epochs, cnn_metrics['Train Loss'], label='CNN Train Loss', linestyle='--', marker='x')
    plt.plot(epochs, cnn_metrics['Validation Loss'], label='CNN Val Loss', linestyle='-', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, resnet_metrics['Train Accuracy'], label='ResNet18 Train Accuracy', linestyle='--', marker='o')
    plt.plot(epochs, resnet_metrics['Validation Accuracy'], label='ResNet18 Val Accuracy', linestyle='-', marker='o')
    plt.plot(epochs, cnn_metrics['Train Accuracy'], label='CNN Train Accuracy', linestyle='--', marker='x')
    plt.plot(epochs, cnn_metrics['Validation Accuracy'], label='CNN Val Accuracy', linestyle='-', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

    # Plot Precision, Recall, and F1-Score
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, resnet_metrics['Precision'], label='ResNet18 Precision', linestyle='--', marker='o')
    plt.plot(epochs, resnet_metrics['Recall'], label='ResNet18 Recall', linestyle='-', marker='o')
    plt.plot(epochs, resnet_metrics['F1-Score'], label='ResNet18 F1-Score', linestyle=':', marker='o')
    plt.plot(epochs, cnn_metrics['Precision'], label='CNN Precision', linestyle='--', marker='x')
    plt.plot(epochs, cnn_metrics['Recall'], label='CNN Recall', linestyle='-', marker='x')
    plt.plot(epochs, cnn_metrics['F1-Score'], label='CNN F1-Score', linestyle=':', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_recall_f1_plot.png')
    plt.show()

    # Plot Training Time per Epoch
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, resnet_metrics['Train Time (s)'], label='ResNet18 Train Time', linestyle='-', marker='o')
    plt.plot(epochs, cnn_metrics['Train Time (s)'], label='CNN Train Time', linestyle='-', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_time_plot.png')
    plt.show()

# Run the plotting function
plot_metrics(resnet_metrics, cnn_metrics)
