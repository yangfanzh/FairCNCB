"""
Fair Graph Neural Network with Counterfactual Augmentation
Implementation for bail prediction task with fairness considerations
"""

# ============== Import Libraries ==============
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, from_networkx

# ============== Data Loading and Preprocessing ==============
# Load bail dataset
data = pd.read_csv('bail.csv')

# Define sensitive attribute (race) and prediction target (recidivism)
sensitive_attr = 'WHITE'  # Binary sensitive attribute (1=white, 0=non-white)
label = 'RECID'           # Binary prediction target (1=recidivism, 0=no recidivism)

# Select feature columns (excluding sensitive attribute and label)
features = [col for col in data.columns if col not in [sensitive_attr, label]]

# Standardize features to zero mean and unit variance
scaler = StandardScaler()
X = scaler.fit_transform(data[features])  # Feature matrix [num_nodes, num_features]
y = data[label].values                    # Target labels
s = data[sensitive_attr].values           # Sensitive attributes

# ============== Graph Construction ==============
def build_knn_graph(X, k=5):
    """
    Construct k-nearest neighbors graph from feature matrix
    Args:
        X: Feature matrix [num_nodes, num_features]
        k: Number of nearest neighbors
    Returns:
        NetworkX graph object
    """
    from sklearn.neighbors import kneighbors_graph
    # Create sparse adjacency matrix using k-nearest neighbors
    adj_matrix = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    # Convert to NetworkX graph
    G = nx.from_scipy_sparse_array(adj_matrix)
    return G

# Build initial graph structure
G = build_knn_graph(X, k=5)

# Convert to PyTorch Geometric data format
data_pyg = from_networkx(G)
data_pyg.x = torch.tensor(X, dtype=torch.float)  # Node features
data_pyg.y = torch.tensor(y, dtype=torch.long)   # Node labels
data_pyg.s = torch.tensor(s, dtype=torch.float)  # Sensitive attributes

# ============== GAN Model for Counterfactual Generation ==============
class Generator(nn.Module):
    """
    Generator network for creating counterfactual node features
    Architecture: Fully connected network with ReLU activations and tanh output
    """
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Tanh activation for bounded output
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs generated features
    Architecture: Fully connected network with LeakyReLU activations and sigmoid output
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),  # LeakyReLU helps prevent gradient vanishing
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability of being real
        )

    def forward(self, x):
        return self.fc(x)

def train_gan(X, s, num_epochs=1000, batch_size=64, latent_dim=100):
    """
    Train GAN to generate counterfactual node features
    Args:
        X: Original node features
        s: Sensitive attributes
        num_epochs: Training epochs
        batch_size: Batch size for training
        latent_dim: Dimension of noise vector for generator
    Returns:
        Trained generator model
    """
    # Prepare real data (features + sensitive attributes)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)  # Add dimension for concatenation
    
    # Combine features and sensitive attributes
    data_tensor = torch.cat([X_tensor, s_tensor], dim=1)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    input_dim = X.shape[1] + 1  # Features + sensitive attribute
    generator = Generator(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Binary cross-entropy loss
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        for real_data in dataloader:
            real_data = real_data[0]
            batch_size = real_data.size(0)
            
            # ===== Train Discriminator =====
            d_optimizer.zero_grad()
            
            # Real data loss
            real_labels = torch.ones(batch_size, 1)  # Real samples labeled 1
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data loss
            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)  # Fake samples labeled 0
            fake_output = discriminator(fake_data.detach())  # Detach to avoid generator update
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ===== Train Generator =====
            g_optimizer.zero_grad()
            
            # Generate new fake data
            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)
            
            # Generator wants discriminator to think these are real
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        # Print training progress
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    return generator

# Train GAN model
generator = train_gan(X, s, num_epochs=500)

def generate_counterfactuals(generator, X, s, num_samples=100, latent_dim=100):
    """
    Generate counterfactual nodes by flipping sensitive attributes
    Args:
        generator: Trained GAN model
        X: Original features
        s: Original sensitive attributes
        num_samples: Number of counterfactuals to generate
        latent_dim: Noise vector dimension
    Returns:
        cf_features: Generated features
        cf_sensitive: Flipped sensitive attributes
    """
    # Generate synthetic data from noise
    noise = torch.randn(num_samples, latent_dim)
    generated_data = generator(noise).detach().numpy()
    
    # Split into features and sensitive attributes
    cf_features = generated_data[:, :-1]  # All columns except last
    cf_sensitive = 1 - generated_data[:, -1].round()  # Flip sensitive attribute
    
    return cf_features, cf_sensitive

# Generate counterfactual nodes (same number as original nodes)
cf_features, cf_sensitive = generate_counterfactuals(generator, X, s, num_samples=len(X))

# ============== Graph Augmentation ==============
def augment_graph(G, X, y, s, cf_features, cf_sensitive):
    """
    Augment original graph with counterfactual nodes
    Args:
        G: Original NetworkX graph
        X: Original features
        y: Original labels
        s: Original sensitive attributes
        cf_features: Counterfactual features
        cf_sensitive: Counterfactual sensitive attributes
    Returns:
        Augmented graph and associated data
    """
    n_original = X.shape[0]  # Number of original nodes
    
    # Combine original and counterfactual data
    X_augmented = np.vstack([X, cf_features])
    y_augmented = np.concatenate([y, np.zeros(len(cf_features))])  # Temporary labels (will be predicted)
    s_augmented = np.concatenate([s, cf_sensitive])
    
    # Create augmented graph
    G_augmented = G.copy()
    
    # Add counterfactual nodes
    for i in range(len(cf_features)):
        G_augmented.add_node(n_original + i)
    
    # Connect counterfactual nodes to original graph using kNN
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    distances, indices = nbrs.kneighbors(cf_features)
    
    # Add edges between counterfactuals and their nearest original nodes
    for i in range(len(cf_features)):
        for j in indices[i]:
            G_augmented.add_edge(n_original + i, j)
    
    return G_augmented, X_augmented, y_augmented, s_augmented

# Augment the original graph
print("Augmenting graph with counterfactual nodes...")
G_augmented, X_augmented, y_augmented, s_augmented = augment_graph(G, X, y, s, cf_features, cf_sensitive)

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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # Only dropout during training
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Classification layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Log probabilities for NLLLoss

# ============== Data Preparation ==============
def prepare_pyg_data(G, X, y):
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    Args:
        G: NetworkX graph
        X: Node features
        y: Node labels
    Returns:
        PyG Data object
    """
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Prepare original and augmented datasets
data_original = prepare_pyg_data(G, X, y)
data_original.s = torch.tensor(s, dtype=torch.float)  # Add sensitive attributes

data_augmented = prepare_pyg_data(G_augmented, X_augmented, y_augmented)
data_augmented.s = torch.tensor(s_augmented, dtype=torch.float)

# ============== Training and Evaluation ==============
def train_and_evaluate(model, data, test_mask=None, epochs=100):
    """
    Train and evaluate GCN model
    Args:
        model: GCN model
        data: PyG Data object
        test_mask: Optional predefined test mask
        epochs: Training epochs
    Returns:
        Dictionary of evaluation metrics
    """
    # Create train/test split if not provided
    if test_mask is None:
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        # Random 80/20 split
        indices = torch.randperm(data.num_nodes)
        train_idx = indices[:int(0.8 * data.num_nodes)]
        test_idx = indices[int(0.8 * data.num_nodes):]
        
        train_mask[train_idx] = True
        test_mask[test_idx] = True
    else:
        train_mask = ~test_mask  # Use complement as training set

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()  # Negative log likelihood loss

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        prob = torch.exp(model(data))[:, 1]  # Probability of class 1
        
        # Standard metrics
        acc = accuracy_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
        f1 = f1_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
        auc = roc_auc_score(data.y[test_mask].numpy(), prob[test_mask].numpy())
        
        # Fairness metrics
        s_test = data.s[test_mask].numpy()  # Sensitive attributes in test set
        y_test = data.y[test_mask].numpy()  # True labels in test set
        pred_test = pred[test_mask].numpy() # Predictions in test set
        
        # Statistical Parity Difference (SPD)
        sp0 = pred_test[s_test == 0].mean()  # Prediction rate for group 0
        sp1 = pred_test[s_test == 1].mean()  # Prediction rate for group 1
        sp_diff = abs(sp0 - sp1)
        
        # Equal Opportunity Difference (EOD)
        eo0 = pred_test[(s_test == 0) & (y_test == 1)].mean()  # True positive rate for group 0
        eo1 = pred_test[(s_test == 1) & (y_test == 1)].mean()  # True positive rate for group 1
        eo_diff = abs(eo0 - eo1)

    return {
        'Accuracy': acc,
        'F1': f1,
        'AUC': auc,
        'SP_diff': sp_diff,  # Statistical Parity Difference (smaller is fairer)
        'EO_diff': eo_diff   # Equal Opportunity Difference (smaller is fairer)
    }

# Initialize models
input_dim = X.shape[1]
hidden_dim = 16  # GCN hidden dimension
output_dim = 2   # Binary classification

# ============== Baseline Model (Original Graph) ==============
model_original = GCN(input_dim, hidden_dim, output_dim)
print("\nTraining on original graph...")
results_original = train_and_evaluate(model_original, data_original)
print("Original Graph Results:", results_original)

# ============== Augmented Model (With Counterfactuals) ==============
# First predict labels for counterfactual nodes using original model
with torch.no_grad():
    cf_pred = model_original(data_augmented).argmax(dim=1)
    data_augmented.y[-len(cf_features):] = cf_pred[-len(cf_features):]  # Update labels

# Train new model on augmented graph
model_augmented = GCN(input_dim, hidden_dim, output_dim)

# Create test mask that only includes original nodes (exclude counterfactuals)
test_mask = torch.zeros(data_augmented.num_nodes, dtype=torch.bool)
test_mask[:len(y)] = True  # Only original nodes are in test set

print("\nTraining on augmented graph...")
results_augmented = train_and_evaluate(model_augmented, data_augmented, test_mask=test_mask)
print("Augmented Graph Results:", results_augmented)
