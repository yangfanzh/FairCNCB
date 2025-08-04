"""
Fair Credit Risk Prediction using Graph Neural Networks with Counterfactual Augmentation
Implementation for German Credit dataset with fairness considerations on gender attribute
"""

# ============== Import Libraries ==============
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

# ============== Data Loading and Preprocessing ==============
# Load German Credit dataset
data = pd.read_csv('German.csv')

# Define sensitive attribute (gender) and prediction target (credit risk)
sensitive_attr = 'Gender'         # Binary sensitive attribute (Male/Female)
target_attr = 'GoodCustomer'      # Binary prediction target (1=good, 0=bad)

# Separate features and labels
features = data.drop(columns=[target_attr])
labels = data[target_attr]

# Identify categorical and numeric columns
categorical_cols = ['PurposeOfLoan']
numeric_cols = [col for col in features.columns if col not in categorical_cols + [sensitive_attr]]

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse=False)
categorical_features = encoder.fit_transform(features[categorical_cols])

# Standardize numeric features
scaler = StandardScaler()
numeric_features = scaler.fit_transform(features[numeric_cols])

# Combine processed features
processed_features = np.hstack([numeric_features, categorical_features])

# Convert gender to binary (Male=1, Female=0)
sensitive_features = (features[sensitive_attr] == 'Male').astype(int).values.reshape(-1, 1)

# Convert to PyTorch tensors
X = torch.FloatTensor(processed_features)
sensitive = torch.FloatTensor(sensitive_features)
y = torch.LongTensor((labels.values + 1) // 2)  # Convert labels to 0/1

# Split data into train and test sets (70/30 split)
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# ============== Graph Construction ==============
# Build k-nearest neighbors graph (k=5)
from sklearn.neighbors import kneighbors_graph
adj_matrix = kneighbors_graph(processed_features, n_neighbors=5, mode='connectivity', include_self=True)

# Convert adjacency matrix to COO format for PyTorch Geometric
adj_coo = adj_matrix.tocoo()
edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)

# Create PyG Data object
data = Data(x=X, edge_index=edge_index, y=y, sensitive=sensitive)

# Create train/test masks
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[:len(X_train)] = True
test_mask[len(X_train):] = True
data.train_mask = train_mask
data.test_mask = test_mask

# ============== GAN Model for Counterfactual Generation ==============
class Generator(nn.Module):
    """
    Generator network for creating counterfactual features
    Architecture: 3-layer fully connected network with ReLU activations
    Takes noise vector and sensitive attribute as input
    """
    def __init__(self, input_dim, output_dim, sensitive_dim=1):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + sensitive_dim, 128),  # Input layer (noise + sensitive)
            nn.ReLU(),                                  # Activation
            nn.Linear(128, 256),                        # Hidden layer
            nn.ReLU(),                                  # Activation
            nn.Linear(256, output_dim)                  # Output layer
        )

    def forward(self, z, sensitive):
        # Concatenate noise and sensitive attribute
        z = torch.cat([z, sensitive], dim=1)
        return self.net(z)

class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs generated features
    Architecture: 3-layer fully connected network with LeakyReLU activations
    """
    def __init__(self, input_dim, sensitive_dim=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + sensitive_dim, 256),  # Input layer
            nn.LeakyReLU(0.2),                          # LeakyReLU helps prevent gradient vanishing
            nn.Linear(256, 128),                        # Hidden layer
            nn.LeakyReLU(0.2),                          # Activation
            nn.Linear(128, 1),                          # Output layer
            nn.Sigmoid()                                # Output probability between 0 and 1
        )

    def forward(self, x, sensitive):
        # Concatenate features and sensitive attribute
        x = torch.cat([x, sensitive], dim=1)
        return self.net(x)

# Initialize GAN parameters
input_dim = processed_features.shape[1]  # Number of features
latent_dim = 100                         # Dimension of noise vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# Initialize generator and discriminator
generator = Generator(latent_dim, input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# Initialize optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Binary cross-entropy loss
criterion = nn.BCELoss()

def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion,
              data_loader, num_epochs=1000, latent_dim=100):
    """
    Train GAN to generate counterfactual features
    Args:
        generator: Generator network
        discriminator: Discriminator network
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        criterion: Loss function
        data_loader: Data loader for training data
        num_epochs: Number of training epochs
        latent_dim: Dimension of noise vector
    """
    for epoch in range(num_epochs):
        for real_data, real_sensitive in data_loader:
            batch_size = real_data.size(0)

            # Move data to device (GPU/CPU)
            real_data = real_data.to(device)
            real_sensitive = real_sensitive.to(device)

            # ===== Train Discriminator =====
            d_optimizer.zero_grad()
            
            # Generate noise
            z = torch.randn(batch_size, latent_dim).to(device)
            
            # Flip sensitive attribute for counterfactuals
            counter_sensitive = 1 - real_sensitive
            
            # Generate fake data
            fake_data = generator(z, counter_sensitive)

            # Real data loss
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_data, real_sensitive)
            d_loss_real = criterion(real_output, real_labels)

            # Fake data loss
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_data.detach(), counter_sensitive)
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ===== Train Generator =====
            g_optimizer.zero_grad()
            
            # Generator tries to fool discriminator
            output = discriminator(fake_data, counter_sensitive)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        # Print training progress every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Prepare data loader for GAN training
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_train, sensitive_train)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train GAN
train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion,
          data_loader, num_epochs=1000, latent_dim=latent_dim)

# ============== Counterfactual Generation ==============
def generate_counterfactuals(generator, num_samples, latent_dim=100):
    """
    Generate counterfactual samples by manipulating gender attribute
    Args:
        generator: Trained generator model
        num_samples: Number of counterfactuals to generate
        latent_dim: Dimension of noise vector
    Returns:
        Tuple of (female-to-male counterfactuals, male-to-female counterfactuals)
    """
    # Generate random noise
    z = torch.randn(num_samples, latent_dim).to(device)

    # Create sensitive attribute vectors
    sensitive_0 = torch.zeros(num_samples, 1).to(device)  # Female
    sensitive_1 = torch.ones(num_samples, 1).to(device)   # Male

    # Generate counterfactuals:
    # Female (0) -> Male (1) counterfactuals
    counter_0_to_1 = generator(z, sensitive_1)
    # Male (1) -> Female (0) counterfactuals
    counter_1_to_0 = generator(z, sensitive_0)

    return counter_0_to_1, counter_1_to_0

# Generate counterfactuals (same number as training samples)
num_samples = X_train.shape[0]
counter_0_to_1, counter_1_to_0 = generate_counterfactuals(generator, num_samples)

# Move to CPU and convert to numpy
counter_0_to_1 = counter_0_to_1.cpu().detach().numpy()
counter_1_to_0 = counter_1_to_0.cpu().detach().numpy()

# ============== Graph Augmentation ==============
def augment_graph(original_features, original_sensitive, counter_0_to_1, counter_1_to_0):
    """
    Augment original graph with counterfactual nodes
    Args:
        original_features: Original feature matrix
        original_sensitive: Original sensitive attributes
        counter_0_to_1: Female-to-male counterfactuals
        counter_1_to_0: Male-to-female counterfactuals
    Returns:
        Tuple of (augmented features, augmented sensitive attributes, augmented labels)
    """
    original_sensitive = original_sensitive.flatten()
    
    # Get indices for each sensitive group
    idx_0 = np.where(original_sensitive == 0)[0]  # Female indices
    idx_1 = np.where(original_sensitive == 1)[0]  # Male indices

    # Randomly select samples to match with counterfactuals
    selected_idx_0 = np.random.choice(idx_0, len(counter_0_to_1), replace=False)
    selected_idx_1 = np.random.choice(idx_1, len(counter_1_to_0), replace=False)

    # Set sensitive attributes for counterfactuals
    counter_sensitive_0_to_1 = np.ones((len(counter_0_to_1), 1))  # Female->Male are now Male (1)
    counter_sensitive_1_to_0 = np.zeros((len(counter_1_to_0), 1))  # Male->Female are now Female (0)

    # Combine original and counterfactual data
    augmented_features = np.vstack([
        original_features,
        counter_0_to_1,
        counter_1_to_0
    ])

    augmented_sensitive = np.vstack([
        original_sensitive.reshape(-1, 1),
        counter_sensitive_0_to_1,
        counter_sensitive_1_to_0
    ])

    # Use original labels for counterfactuals (from their source nodes)
    augmented_labels = np.concatenate([
        y.numpy(),
        y.numpy()[selected_idx_0],  # Labels from original female samples
        y.numpy()[selected_idx_1]   # Labels from original male samples
    ])

    return augmented_features, augmented_sensitive, augmented_labels

# Augment the graph with counterfactuals
augmented_features, augmented_sensitive, augmented_labels = augment_graph(
    processed_features, sensitive.numpy(), counter_0_to_1, counter_1_to_0
)

# Build new adjacency matrix for augmented graph
augmented_adj = kneighbors_graph(augmented_features, n_neighbors=5, mode='connectivity', include_self=True)
augmented_coo = augmented_adj.tocoo()
augmented_edge_index = torch.tensor([augmented_coo.row, augmented_coo.col], dtype=torch.long)

# Create augmented PyG Data object
augmented_data = Data(
    x=torch.FloatTensor(augmented_features),
    edge_index=augmented_edge_index,
    y=torch.LongTensor(augmented_labels),
    sensitive=torch.FloatTensor(augmented_sensitive)
)

# Create train/test masks for augmented data
train_size = len(X_train)
test_size = len(X_test)
augmented_train_mask = torch.zeros(augmented_data.num_nodes, dtype=torch.bool)
augmented_test_mask = torch.zeros(augmented_data.num_nodes, dtype=torch.bool)

# Original training nodes + counterfactuals are in train set
augmented_train_mask[:train_size + 2 * num_samples] = True
# Original test nodes are in test set
augmented_test_mask[train_size:train_size + test_size] = True

augmented_data.train_mask = augmented_train_mask
augmented_data.test_mask = augmented_test_mask

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
        self.dropout = nn.Dropout(0.5)                # Dropout for regularization

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Final classification layer
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)  # Log probabilities for NLLLoss

# Initialize GCN model
input_dim = augmented_data.num_features
hidden_dim = 64
output_dim = 2  # Binary classification

model = GCN(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()  # Negative log likelihood loss

# ============== Training and Evaluation Functions ==============
def train(model, data):
    """Train GCN model for one epoch"""
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y.to(device)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    """Evaluate GCN model on test set"""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        
        # Overall accuracy
        acc = accuracy_score(data.y[data.test_mask].numpy(), pred[data.test_mask].cpu().numpy())

        # Calculate accuracy by sensitive group
        test_indices = torch.where(data.test_mask)[0].numpy()
        sensitive_test = data.sensitive[test_indices].numpy().flatten()
        y_test = data.y[test_indices].numpy()
        pred_test = pred[test_indices].cpu().numpy()
        
        # Accuracy for female group (sensitive=0)
        mask_0 = sensitive_test == 0
        acc_0 = accuracy_score(y_test[mask_0], pred_test[mask_0]) if sum(mask_0) > 0 else 0
        
        # Accuracy for male group (sensitive=1)
        mask_1 = sensitive_test == 1
        acc_1 = accuracy_score(y_test[mask_1], pred_test[mask_1]) if sum(mask_1) > 0 else 0

        # Fairness metric (absolute difference in accuracy between groups)
        fair_metric = abs(acc_0 - acc_1)

    return acc, fair_metric

# Train augmented model
print("\nTraining augmented model (with counterfactuals)...")
for epoch in range(200):
    loss = train(model, augmented_data)
    if epoch % 20 == 0:
        acc, fair_metric = test(model, augmented_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Fairness: {fair_metric:.4f}')

def evaluate_model(model, data):
    """
    Comprehensive evaluation of model performance and fairness
    Args:
        model: Trained GCN model
        data: PyG Data object with test_mask
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)

        # Get test indices
        test_indices = torch.where(data.test_mask)[0].numpy()

        # Get true and predicted labels
        y_true = data.y[test_indices].numpy()
        y_pred = pred[test_indices].cpu().numpy()

        # Get sensitive attributes
        sensitive = data.sensitive[test_indices].numpy().flatten()

        # Overall metrics
        overall_acc = accuracy_score(y_true, y_pred)
        overall_f1 = f1_score(y_true, y_pred)

        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each sensitive group
        for s in [0, 1]:  # 0=Female, 1=Male
            mask = sensitive == s
            if sum(mask) == 0:  # Skip if no samples in this group
                continue

            y_true_s = y_true[mask]
            y_pred_s = y_pred[mask]

            # Group-specific metrics
            results[f'acc_s{s}'] = accuracy_score(y_true_s, y_pred_s)
            results[f'f1_s{s}'] = f1_score(y_true_s, y_pred_s)
            results[f'tpr_s{s}'] = np.sum((y_true_s == 1) & (y_pred_s == 1)) / np.sum(y_true_s == 1)  # True Positive Rate
            results[f'fpr_s{s}'] = np.sum((y_true_s == 0) & (y_pred_s == 1)) / np.sum(y_true_s == 0)  # False Positive Rate

        # Calculate fairness metrics (differences between groups)
        if 'acc_s0' in results and 'acc_s1' in results:
            results['acc_diff'] = abs(results['acc_s0'] - results['acc_s1'])
        if 'tpr_s0' in results and 'tpr_s1' in results:
            results['tpr_diff'] = abs(results['tpr_s0'] - results['tpr_s1'])  # Equal Opportunity Difference
        if 'fpr_s0' in results and 'fpr_s1' in results:
            results['fpr_diff'] = abs(results['fpr_s0'] - results['fpr_s1'])  # Equalized Odds Difference

        # Add overall metrics
        results['overall_acc'] = overall_acc
        results['overall_f1'] = overall_f1

        return results

# Evaluate augmented model
print("\nEvaluating augmented model...")
fairness_results = evaluate_model(model, augmented_data)
print("Fairness Evaluation Results:")
for k, v in fairness_results.items():
    print(f"{k}: {v:.4f}")

# ============== Baseline Model (Without Counterfactuals) ==============
# Initialize baseline model
original_model = GCN(input_dim, hidden_dim, output_dim).to(device)
original_optimizer = optim.Adam(original_model.parameters(), lr=0.01, weight_decay=5e-4)

# Create Data object for original (non-augmented) graph
original_data = Data(
    x=X,
    edge_index=edge_index,
    y=y,
    sensitive=sensitive
)

# Create train/test masks for original data
original_train_mask = torch.zeros(original_data.num_nodes, dtype=torch.bool)
original_test_mask = torch.zeros(original_data.num_nodes, dtype=torch.bool)
original_train_mask[:len(X_train)] = True
original_test_mask[len(X_train):] = True
original_data.train_mask = original_train_mask
original_data.test_mask = original_test_mask

# Train baseline model
print("\nTraining baseline model (without counterfactuals)...")
for epoch in range(200):
    loss = train(original_model, original_data)
    if epoch % 20 == 0:
        acc, fair_metric = test(original_model, original_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Fairness: {fair_metric:.4f}')

# Evaluate baseline model
print("\nEvaluating baseline model...")
original_results = evaluate_model(original_model, original_data)
for k, v in original_results.items():
    print(f"{k}: {v:.4f}")
