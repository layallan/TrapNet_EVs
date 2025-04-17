import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from KAN import KANLinear  # Import the KANLinear layer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score
import math
import random
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

seed = 1998618
set_seed(seed)

# Load the training data
data = scipy.io.loadmat('./data/data_new.mat')['data_new']
numObservations = data.shape[1]  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

features = []
labels = []

for i in range(numObservations):
    element = data[0, i]
    original_feature = element[0, :]  
    label = element[1, 0]            
    features.append(original_feature)
    labels.append(label)
    
# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long).squeeze() - 1  

print("Features shape:", features.shape) 
print("Labels shape:", labels.shape) 
print("Unique labels:", torch.unique(labels))  

dataset = TensorDataset(features, labels)

# Spli training and validation sets (80-20 split)
train_size = int(numObservations * 0.8)
val_size = int(numObservations * 0.1)#0.1
test_size = numObservations - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print("Train size:", train_size)  
print("val_size:", val_size)  
print("test_size:", test_size)  

batch_size_n = 6#6 #8
train_loader = DataLoader(train_dataset, batch_size=batch_size_n, shuffle=True) #16
val_loader = DataLoader(val_dataset, batch_size=batch_size_n, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size_n, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding='same')
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding='same')
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(512, 2048, kernel_size=3, padding=1)#7
        # self.conv5 = nn.Conv1d(1024, 2048, kernel_size=7, padding=1)#7
        
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(512)
        self.norm4 = nn.LayerNorm(2048)
        # self.norm5 = nn.LayerNorm(2048)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = KANLinear(in_features=2048, out_features=3,grid_size=1,
        spline_order=5,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x, return_latent=False):
        x = x.unsqueeze(1)  # Adding channel dimension
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm expects (batch_size, sequence_length, num_features)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.norm4(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        # x = self.conv5(x)
        # x = self.norm5(x.transpose(1, 2)).transpose(1, 2)
        # x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        latent_representation = x
        x = self.fc(x)
        if return_latent:
            return x, latent_representation
        else:
            return x
        # return x


# Initialize and move the model to the device (GPU/CPU)
model = ConvNet().to(device)

# Training parameters
criterion = nn.CrossEntropyLoss()

# criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0) # 1e-4

# scheduler=lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0)

num_epochs = 8000
early_stop_patience = 200 #40
best_val_loss = float('inf')
patience_counter = 0

best_val_loss = float('inf')
best_val_accuracy = 0.0


# Lists to store loss and accuracy
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []



######### reture latent space features
from sklearn.manifold import TSNE

def extract_latent(loader):
    model.eval()
    latent_space = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, latent = model(inputs, return_latent=True)
            latent_space.append(latent.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(latent_space), np.concatenate(labels_list)

# Extract latent spaces for train and test sets
train_latent, train_labels = extract_latent(train_loader)
test_latent, test_labels = extract_latent(test_loader)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=seed)
train_tsne_results = tsne.fit_transform(train_latent)
test_tsne_results = tsne.fit_transform(test_latent)

# Plot t-SNE results
def plot_tsne(tsne_results, labels, title, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')

    unique_labels = np.unique(labels)
    
    handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.viridis(i / len(unique_labels)),
                          label=f'Label {int(label)}', markersize=8, linestyle='None') 
               for i, label in enumerate(unique_labels)]

    plt.legend(handles=handles, title="Labels", loc="best")
    # plt.grid(True)
    plt.savefig(filename)  
    plt.show()



model.load_state_dict(torch.load('./model/best_model_by_accuracy.pth'))

# Test the model
model.eval()
correct = 0
total = 0
test_losses = []
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_accuracy = correct / total
average_test_loss = sum(test_losses) / len(test_losses)
# print(f'[accuracy priority] Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
f1 = f1_score(all_labels, all_predictions, average='weighted') 
recall = recall_score(all_labels, all_predictions, average='weighted')

print(f'[accuracy priority] Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}')

print('Finished Training and Testing')

cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_best_acc.pdf')
plt.show()

train_latent, train_labels = extract_latent(train_loader)
test_latent, test_labels = extract_latent(test_loader)


tsne = TSNE(n_components=2, random_state=seed)
train_tsne_results = tsne.fit_transform(train_latent)
test_tsne_results = tsne.fit_transform(test_latent)

plot_tsne(train_tsne_results, train_labels, 'Train Latent Space Visualization', 'train_tsne_results_acc.pdf')
plot_tsne(test_tsne_results, test_labels, 'Test Latent Space Visualization', 'test_tsne_results_acc.pdf')