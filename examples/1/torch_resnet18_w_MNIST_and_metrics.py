import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
import pandas as pd

# Check if MPS (Metal Performance Shaders) is available and set the device accordingly
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# MNIST dataset
train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Data loaders with increased number of workers for faster data loading
loaders = {
    'train': DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4, pin_memory=True),
    'test': DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True),
}

# Load ResNet18 model
resnet18 = models.resnet18(pretrained=False)

# Modify the first convolutional layer to accept grayscale images
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the output layer to match the number of classes in MNIST
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

# Move the model to the device
resnet18 = resnet18.to(device)

# Loss and optimizer with learning rate adjustment
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

# Scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=device.type != 'cpu')


# Training function
def train(num_epochs, model, loaders):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
                outputs = model(images)
                loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(loaders['train'])
        train_accuracy = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')


# Testing function with additional metrics
def test(model):
    model.eval()
    y_true = []
    y_pred = []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    inference_time = time.time() - start_time
    accuracy = sum([y_pred[i] == y_true[i] for i in range(len(y_true))]) / len(y_true) * 100

    # Save metrics to a file
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'Inference Time (s)': [inference_time]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('model_metrics.csv', index=False)

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
    print(f'Inference Time: {inference_time:.4f} seconds')


# Train and evaluate the model
num_epochs = 10
train(num_epochs, resnet18, loaders)
test(resnet18)