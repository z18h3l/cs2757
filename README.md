java cLab 2: Neural Networks for Image 
Classification
Duration: 2 hours
Tools:
• Jupyter Notebook
• IDE: PyCharm==2024.2.3 (or any IDE of your choice)
• Python: 3.12
• Libraries:
o PyTorch==2.4.0
o TorchVision==0.19.0
o Matplotlib==3.9.2
Learning Objectives:
• Understand the basic architecture of a neural network.
• Load and explore the CIFAR-10 dataset.
• Implement and train a neural network, individualized by your QMUL ID.
• Verify machine learning concepts such as accuracy, loss, and evaluation metrics 
by running predefined code.
Lab Outline:
In this lab, you will implement a simple neural network model to classify images from 
the CIFAR-10 dataset. The task will be individualized based on your QMUL ID to ensure 
unique configurations for each student.
1. Task 1: Understanding the CIFAR-10 Dataset
• The CIFAR-10 dataset consists of 60,000 32x32 color images categorized into 10 
classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks).
• The dataset is divided into 50,000 training images and 10,000 testing images.
• You will load the CIFAR-10 dataset using PyTorch’s built-in torchvision library.
Step-by-step Instructions:
1. Open the provided Jupyter Notebook.
2. Load and explore the CIFAR-10 dataset using the following code:
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# Basic transformations for the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), 
transforms.Normalize((0.5,), (0.5,))])
# Load the CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, 
download=True, transform=transform)
2. Task 2: Individualized Neural Network Implementation, Training, and Test
You will implement a neural network model to classify images from the CIFAR-10 
dataset. However, certain parts of the task will be individualized based on your QMUL 
ID. Follow the instructions carefully to ensure your model’s configuration is unique.
Step 1: Dataset Split Based on Your QMUL ID
You will use the last digit of your QMUL ID to define the training-validation split:
• If your ID ends in 0-4: use a 70-30 split (70% training, 30% validation).
• If your ID ends in 5-9: use an 80-20 split (80% training, 20% validation).
Code:
from torch.utils.data import random_split
# Set the student's last digit of the ID (replace with 
your own last digit)
last_digit_of_id = 7 # Example: Replace this with the 
last digit of your QMUL ID
# Define the split ratio based on QMUL ID
split_ratio = 0.7 if last_digit_of_id <= 4 else 0.8
# Split the dataset
train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, 
[train_size, val_size])
# DataLoaders
from torch.utils.data import DataLoader
batch_size = 32 + last_digit_of_id # Batch size is 32 + 
last digit of your QMUL ID
train_loader = DataLoader(train_dataset, 
batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, 
batch_size=batch_size, shuffle=False)
print(f"Training on {train_size} images, Validating on 
{val_size} images.")
Step 2: Predefined Neural Network Model
You will use a predefined neural network architecture provided in the lab. The model’s 
hyperparameters will be customized based on your QMUL ID.
1. Learning Rate: Set the learning rate to 0.001 + (last digit of your QMUL ID * 
0.0001).
2. Number of Epochs: Train your model for 10 + (last digit of your QMUL ID) 
epochs.
Code:
import torch
import torch.optim as optim
# Define the model
model = torch.nn.Sequential(
 torch.nn.Flatten(),
 torch.nn.Linear(32*32*3, 512),
 torch.nn.ReLU(),
 torch.nn.Linear(512, 10) # 10 output classes for 
CIFAR-10
)
# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# Learning rate based on QMUL ID
learning_rate = 0.001 + (last_digit_of_id * 0.0001)
optimizer = optim.A代 写data、Python
代做程序编程语言dam(model.parameters(), 
lr=learning_rate)
# Number of epochs based on QMUL ID
num_epochs = 100 + last_digit_of_id
print(f"Training for {num_epochs} epochs with learning 
rate {learning_rate}.")
Step 3: Model Training and Evaluation
Use the provided training loop to train your model and evaluate it on the validation set. 
Track the loss and accuracy during the training process.
Expected Output: For training with around 100 epochs, it may take 0.5~1 hour to finish. 
You may see a lower accuracy, especially for the validation accuracy, due to the lower 
number of epochs or the used simple neural network model, etc. If you are interested, 
you can find more advanced open-sourced codes to test and improve the performance. 
In this case, it may require a long training time on the CPU-based device.
Code:
# Training loop
train_losses = [] 
train_accuracies = []
val_accuracies = []
for epoch in range(num_epochs):
 model.train()
 running_loss = 0.0
 correct = 0
 total = 0
 for inputs, labels in train_loader:
 optimizer.zero_grad()
 outputs = model(inputs)
 loss = criterion(outputs, labels)
 loss.backward()
 optimizer.step()
 
 running_loss += loss.item()
 _, predicted = torch.max(outputs, 1)
 total += labels.size(0)
 correct += (predicted == labels).sum().item()
 train_accuracy = 100 * correct / total
 print(f"Epoch {epoch+1}/{num_epochs}, Loss: 
{running_loss:.4f}, Training Accuracy: 
{train_accuracy:.2f}%")
 
 # Validation step
 model.eval()
 correct = 0
 total = 0
 with torch.no_grad():
 for inputs, labels in val_loader:
 outputs = model(inputs)
 _, predicted = torch.max(outputs, 1)
 total += labels.size(0)
 correct += (predicted == labels).sum().item()
 
 val_accuracy = 100 * correct / total
 print(f"Validation Accuracy after Epoch {epoch + 1}: 
{val_accuracy:.2f}%")
 train_losses.append(running_loss) 
 train_accuracies.append(train_accuracy)
 val_accuracies.append(val_accuracy)
Task 3: Visualizing and Analyzing the Results
Visualize the results of the training and validation process. Generate the following plots 
using Matplotlib:
• Training Loss vs. Epochs.
• Training and Validation Accuracy vs. Epochs.
Code for Visualization:
import matplotlib.pyplot as plt
# Plot Loss
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, 
label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()
# Plot Accuracy
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, 
label="Training Accuracy")
plt.plot(range(1, num_epochs + 1), val_accuracies, 
label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()
Lab Report Submission and Marking Criteria
After completing the lab, you need to submit a report that includes:
1. Individualized Setup (20/100):
o Clearly state the unique configurations used based on your QMUL ID, 
including dataset split, number of epochs, learning rate, and batch size.
2. Neural Network Architecture and Training (30/100):
o Provide an explanation of the model architecture (i.e., the number of input 
layer, hidden layer, and output layer, activation function) and training 
procedure (i.e., the used optimizer).
o Include the plots of training loss, training and validation accuracy.
3. Results Analysis (30/100):
o Provide analysis of the training and validation performance.
o Reflect on whether the model is overfitting or underfitting based on the 
provided results.
4. Concept Verification (20/100):
o Answer the provided questions below regarding machine learning 
concepts.
(1) What is overfitting issue? List TWO methods for addressing the overfitting 
issue.
(2) What is the role of loss function? List TWO representative loss functions.

         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
