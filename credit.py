"""
Fair Credit Risk Prediction using Graph Neural Networks with Counterfactual Augmentation
Implementation for credit default prediction with fairness considerations
"""

# ============== Import Libraries ==============
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============== Data Loading and Preprocessing ==============
# Load credit dataset
data = pd.read_csv('credit.csv')

# Define sensitive attribute (marital status) and prediction target (default)
sensitive_attr = 'Married'          # Binary sensitive attribute (1=married, 0=unmarried)
target_attr = 'NoDefaultNextMonth'  # Binary prediction target (1=no default, 0=default)

# Separate features and target
features = data.drop(columns=[target_attr])
target = data[target_attr]

# Define continuous and categorical features
continuous_features = [
    'Age', 'MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months',
    'MonthsWithZeroBalanceOverLast6Months', 'MonthsWithLowSpendingOverLast6Months',
    'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount',
    'MostRecentPaymentAmount', 'TotalOverdueCounts', 'TotalMonthsOverdue'
]

categorical_features = [
    'EducationLevel', 
    'HistoryOfOverduePayments'
]

# Standardize continuous features
scaler = StandardScaler()
features[continuous_features] = scaler.fit_transform(features[continuous_features])

# One-hot encode categorical features
features = pd.get_dummies(features, columns=categorical_features)

# Split data into train and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)  # Add extra dimension
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

# ============== GAN Model for Counterfactual Generation ==============
class Generator(nn.Module):
    """
    Generator network for creating counterfactual features
    Architecture: 3-layer fully connected network with ReLU activations and tanh output
    """
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # Input layer
            nn.ReLU(),                  # Activation
            nn.Linear(128, 256),        # Hidden layer
            nn.ReLU(),                  # Activation
            nn.Linear(256, output_dim), # Output layer
            nn.Tanh()                   # Tanh activation bounds output to [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs generated features
    Architecture: 3-layer fully connected network with LeakyReLU activations
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # Input layer
            nn.LeakyReLU(0.2),         # LeakyReLU helps prevent gradient vanishing
            nn.Linear(256, 128),        # Hidden layer
            nn.LeakyReLU(0.2),          # Activation
            nn.Linear(128, 1),          # Output layer
            nn.Sigmoid()                # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

# Initialize GAN parameters
input_dim = X_train.shape[1]  # Number of features
latent_dim = 100              # Dimension of noise vector for generator
batch_size = 64               # Batch size for training
epochs = 500                  # Number of training epochs
lr = 0.0002                   # Learning rate

# Initialize generator and discriminator
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)

# Initialize optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)  # Generator optimizer
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)  # Discriminator optimizer

# Binary cross-entropy loss
criterion = nn.BCELoss()

# Create dataloader for training
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# GAN Training Loop
for epoch in range(epochs):
    for i, (real_data, _) in enumerate(dataloader):
        batch_size = real_data.size(0)
        
        # ===== Train Discriminator =====
        optimizer_D.zero_grad()
        
        # Real data loss
        real_labels = torch.ones(batch_size, 1)  # Real samples labeled 1
        output = discriminator(real_data)
        d_loss_real = criterion(output, real_labels)
        
        # Fake data loss
        noise = torch.randn(batch_size, latent_dim)  # Generate random noise
        fake_data = generator(noise)                # Generate fake samples
        fake_labels = torch.zeros(batch_size, 1)     # Fake samples labeled 0
        output = discriminator(fake_data.detach())   # Detach to avoid generator update
        d_loss_fake = criterion(output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # ===== Train Generator =====
        optimizer_G.zero_grad()
        
        # Generate new fake data
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        output = discriminator(fake_data)
        
        # Generator tries to fool discriminator (labels as real)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    # Print training progress every 50 epochs
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# ============== Counterfactual Generation ==============
def generate_counterfactuals(generator, original_data, sensitive_idx, target_value, num_samples):
    """
    Generate counterfactual samples by manipulating sensitive attribute
    Args:
        generator: Trained generator model
        original_data: Original feature tensor
        sensitive_idx: Index of sensitive attribute column
        target_value: Desired value for sensitive attribute (0 or 1)
        num_samples: Number of counterfactuals to generate
    Returns:
        Tensor of generated counterfactual samples
    """
    # Generate random noise
    noise = torch.randn(num_samples, latent_dim)
    
    # Generate synthetic samples
    synthetic_samples = generator(noise).detach()
    
    # Set sensitive attribute to target value
    synthetic_samples[:, sensitive_idx] = target_value
    
    # Clip values to [-1, 1] to match tanh activation range
    synthetic_samples = torch.clamp(synthetic_samples, -1, 1)
    
    return synthetic_samples

# Get index of sensitive attribute column
sensitive_idx = features.columns.get_loc(sensitive_attr)

# Number of counterfactuals to generate for each group
num_counterfactuals = 500

# Generate counterfactuals for married (1) and unmarried (0) groups
counterfactuals_married = generate_counterfactuals(generator, X_train_tensor, sensitive_idx, 1, num_counterfactuals)
counterfactuals_unmarried = generate_counterfactuals(generator, X_train_tensor, sensitive_idx, 0, num_counterfactuals)

# Combine original and counterfactual data
X_train_extended = torch.cat([
    X_train_tensor, 
    counterfactuals_married, 
    counterfactuals_unmarried
], dim=0)

# Create labels for counterfactuals (assuming same distribution as training data)
y_counterfactuals = torch.cat([
    torch.ones(num_counterfactuals, 1),  # Married counterfactuals
    torch.zeros(num_counterfactuals, 1)   # Unmarried counterfactuals
])

# Combine original and counterfactual labels
y_train_extended = torch.cat([y_train_tensor, y_counterfactuals], dim=0)

# ============== Graph Construction ==============
def build_graph(features, k=5):
    """
    Build k-nearest neighbors graph from feature matrix
    Args:
        features: Feature tensor [num_nodes, num_features]
        k: Number of nearest neighbors
    Returns:
        Adjacency matrix for the graph
    """
    features_np = features.numpy()
    
    # Compute pairwise Euclidean distances
    distances = np.zeros((features_np.shape[0], features_np.shape[0]))
    for i in range(features_np.shape[0]):
        for j in range(features_np.shape[0]):
            distances[i, j] = np.linalg.norm(features_np[i] - features_np[j])
    
    # Create adjacency matrix by connecting to k-nearest neighbors
    adj = np.zeros_like(distances)
    for i in range(distances.shape[0]):
        idx = np.argpartition(distances[i], k)[:k + 1]  # Get indices of k nearest neighbors
        adj[i, idx] = 1  # Create undirected edges
        adj[idx, i] = 1
    
    # Remove self-loops
    np.fill_diagonal(adj, 0)
    
    return adj

# Build graph from extended training data
adj_matrix = build_graph(X_train_extended)

# Convert adjacency matrix to edge index format (PyG format)
edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long)

# Create PyG Data object
graph_data = Data(
    x=X_train_extended,      # Node features
    edge_index=edge_index,   # Graph connectivity
    y=y_train_extended       # Node labels
)

# ============== GCN Model Definition ==============
class GCN(nn.Module):
    """
    Graph Convolutional Network for node classification
    Architecture: Two GCN layers + linear classifier
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First graph convolution
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second graph convolution
        self.fc = nn.Linear(hidden_dim, output_dim)   # Final classification layer
        self.sigmoid = nn.Sigmoid()                  # Sigmoid activation for binary classification

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Final classification layer
        x = self.fc(x)
        return self.sigmoid(x)  # Output probabilities between 0 and 1

# Initialize GCN model
input_dim = X_train_extended.shape[1]  # Number of features
hidden_dim = 64                        # Hidden layer dimension
output_dim = 1                         # Binary classification output

model = GCN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()  # Binary cross-entropy loss

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(graph_data.x, graph_data.edge_index)
    
    # Compute loss
    loss = criterion(outputs, graph_data.y)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print training progress every 20 epochs
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# ============== Evaluation ==============
def evaluate(model, X, y, sensitive_attr, sensitive_idx):
    """
    Evaluate model performance and fairness metrics
    Args:
        model: Trained GCN model
        X: Test features
        y: Test labels
        sensitive_attr: Name of sensitive attribute
        sensitive_idx: Column index of sensitive attribute
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        # Build graph for test data
        test_adj = build_graph(X, k=5)
        test_edge_index = torch.tensor(np.array(np.where(test_adj)), dtype=torch.long)
        test_data = Data(x=X, edge_index=test_edge_index)

        # Get model predictions
        outputs = model(test_data.x, test_data.edge_index)
        predictions = (outputs > 0.5).float()  # Threshold at 0.5
        
        # Convert to numpy arrays
        y_np = y.numpy().flatten()
        pred_np = predictions.numpy().flatten()
        prob_np = outputs.numpy().flatten()
        sensitive_np = X[:, sensitive_idx].numpy().flatten()

        # Standard metrics
        acc = accuracy_score(y_np, pred_np)
        f1 = f1_score(y_np, pred_np)
        auc = roc_auc_score(y_np, prob_np)

        # Fairness metrics
        # Statistical Parity Difference (SPD)
        sp0 = np.mean(pred_np[sensitive_np == 0]) - np.mean(pred_np)
        sp1 = np.mean(pred_np[sensitive_np == 1]) - np.mean(pred_np)
        sp = max(abs(sp0), abs(sp1))  # Maximum deviation from overall mean

        # Equal Opportunity Difference (EOD)
        y1_idx = y_np == 1  # Indices of positive class
        eo0 = np.mean(pred_np[(sensitive_np == 0) & y1_idx]) - np.mean(pred_np[y1_idx])
        eo1 = np.mean(pred_np[(sensitive_np == 1) & y1_idx]) - np.mean(pred_np[y1_idx])
        eo = max(abs(eo0), abs(eo1))  # Maximum deviation from positive class mean

        return {
            'Accuracy': acc,
            'F1': f1,
            'AUC': auc,
            'Statistical Parity': sp,
            'Equal Opportunity': eo
        }

# Evaluate model on test set
print("\nEvaluating augmented model...")
test_results = evaluate(model, X_test_tensor, y_test_tensor, sensitive_attr, sensitive_idx)

# Print evaluation metrics
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# ============== Baseline Model (Without Counterfactuals) ==============
# Build graph from original training data
orig_adj_matrix = build_graph(X_train_tensor)
orig_edge_index = torch.tensor(np.array(np.where(orig_adj_matrix)), dtype=torch.long)
orig_graph_data = Data(x=X_train_tensor, edge_index=orig_edge_index, y=y_train_tensor)

# Initialize baseline model
model_orig = GCN(input_dim, hidden_dim, output_dim)
optimizer_orig = optim.Adam(model_orig.parameters(), lr=0.01)

# Train baseline model
print("\nTraining baseline model (without counterfactuals)...")
for epoch in range(num_epochs):
    model_orig.train()
    optimizer_orig.zero_grad()
    outputs = model_orig(orig_graph_data.x, orig_graph_data.edge_index)
    loss = criterion(outputs, orig_graph_data.y)
    loss.backward()
    optimizer_orig.step()

# Evaluate baseline model
print("\nEvaluating baseline model...")
orig_test_results = evaluate(model_orig, X_test_tensor, y_test_tensor, sensitive_attr, sensitive_idx)

# Print baseline results
for metric, value in orig_test_results.items():
    print(f"{metric}: {value:.4f}")
