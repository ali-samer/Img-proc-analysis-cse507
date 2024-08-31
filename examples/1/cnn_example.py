import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import time

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset
train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Data loaders
loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0, pin_memory=True),
    'test': DataLoader(test_data, batch_size=100, shuffle=False, num_workers=0, pin_memory=True),
}


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output
        output = self.out(x)
        return output, x  # return x for visualization


# Instantiate the model, loss function, and optimizer
cnn = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)

# Initialize metrics log
metrics_log = []


# Training function
def train(num_epochs, model, loaders):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # Training phase
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(images)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(loaders['train'])
        train_accuracy = 100 * correct / total
        train_time = time.time() - start_time

        # Validation phase
        val_loss, val_accuracy, precision, recall, f1 = validate(model, loaders['test'])

        # Log metrics
        metrics_log.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Accuracy': train_accuracy,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Train Time (s)': train_time
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
              f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, "
              f"Train Time: {train_time:.2f}s")


# Validation function to compute loss and accuracy on the validation set
def validate(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    return val_loss, val_accuracy, precision, recall, f1


# Save metrics log to CSV for comparison with other models
def save_metrics():
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv('cnn_model_metrics_detailed.csv', index=False)
    print("Metrics saved to 'cnn_model_metrics_detailed.csv'")


# Ensure main block is used when running multiprocessing code
if __name__ == '__main__':
    num_epochs = 50
    train(num_epochs, cnn, loaders)
    save_metrics()
