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


data = pd.read_csv('bail.csv')

sensitive_attr = 'WHITE'
label = 'RECID'
features = [col for col in data.columns if col not in [sensitive_attr, label]]

scaler = StandardScaler()
X = scaler.fit_transform(data[features])
y = data[label].values
s = data[sensitive_attr].values

def build_knn_graph(X, k=5):
    from sklearn.neighbors import kneighbors_graph
    adj_matrix = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    G = nx.from_scipy_sparse_array(adj_matrix)
    return G


G = build_knn_graph(X, k=5)
data_pyg = from_networkx(G)
data_pyg.x = torch.tensor(X, dtype=torch.float)
data_pyg.y = torch.tensor(y, dtype=torch.long)
data_pyg.s = torch.tensor(s, dtype=torch.float)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

def train_gan(X, s, num_epochs=1000, batch_size=64, latent_dim=100):

    X_tensor = torch.tensor(X, dtype=torch.float32)
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)

    data_tensor = torch.cat([X_tensor, s_tensor], dim=1)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1] + 1
    generator = Generator(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for real_data in dataloader:
            real_data = real_data[0]
            batch_size = real_data.size(0)
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            g_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    return generator

generator = train_gan(X, s, num_epochs=500)

def generate_counterfactuals(generator, X, s, num_samples=100, latent_dim=100):

    noise = torch.randn(num_samples, latent_dim)
    generated_data = generator(noise).detach().numpy()

    cf_features = generated_data[:, :-1]
    cf_sensitive = 1 - generated_data[:, -1].round()

    return cf_features, cf_sensitive

cf_features, cf_sensitive = generate_counterfactuals(generator, X, s, num_samples=len(X))

def augment_graph(G, X, y, s, cf_features, cf_sensitive):

    n_original = X.shape[0]

    X_augmented = np.vstack([X, cf_features])
    y_augmented = np.concatenate([y, np.zeros(len(cf_features))])
    s_augmented = np.concatenate([s, cf_sensitive])


    G_augmented = G.copy()


    for i in range(len(cf_features)):
        G_augmented.add_node(n_original + i)


    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    distances, indices = nbrs.kneighbors(cf_features)

    for i in range(len(cf_features)):
        for j in indices[i]:
            G_augmented.add_edge(n_original + i, j)

    return G_augmented, X_augmented, y_augmented, s_augmented


print("Augmenting graph with counterfactual nodes...")
G_augmented, X_augmented, y_augmented, s_augmented = augment_graph(G, X, y, s, cf_features, cf_sensitive)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)
def prepare_pyg_data(G, X, y):

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

data_original = prepare_pyg_data(G, X, y)
data_original.s = torch.tensor(s, dtype=torch.float)

data_augmented = prepare_pyg_data(G_augmented, X_augmented, y_augmented)
data_augmented.s = torch.tensor(s_augmented, dtype=torch.float)

def train_and_evaluate(model, data, test_mask=None, epochs=100):
    if test_mask is None:

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        indices = torch.randperm(data.num_nodes)
        train_idx = indices[:int(0.8 * data.num_nodes)]
        test_idx = indices[int(0.8 * data.num_nodes):]

        train_mask[train_idx] = True
        test_mask[test_idx] = True
    else:

        train_mask = ~test_mask

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        prob = torch.exp(model(data))[:, 1]
        acc = accuracy_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
        f1 = f1_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
        auc = roc_auc_score(data.y[test_mask].numpy(), prob[test_mask].numpy())

        s_test = data.s[test_mask].numpy()
        y_test = data.y[test_mask].numpy()
        pred_test = pred[test_mask].numpy()

        sp0 = pred_test[s_test == 0].mean()
        sp1 = pred_test[s_test == 1].mean()
        sp_diff = abs(sp0 - sp1)

        eo0 = pred_test[(s_test == 0) & (y_test == 1)].mean()
        eo1 = pred_test[(s_test == 1) & (y_test == 1)].mean()
        eo_diff = abs(eo0 - eo1)

    return {
        'Accuracy': acc,
        'F1': f1,
        'AUC': auc,
        'SP_diff': sp_diff,
        'EO_diff': eo_diff
    }



input_dim = X.shape[1]
hidden_dim = 16
output_dim = 2

model_original = GCN(input_dim, hidden_dim, output_dim)
results_original = train_and_evaluate(model_original, data_original)


with torch.no_grad():
    cf_pred = model_original(data_augmented).argmax(dim=1)
    data_augmented.y[-len(cf_features):] = cf_pred[-len(cf_features):]


model_augmented = GCN(input_dim, hidden_dim, output_dim)


test_mask = torch.zeros(data_augmented.num_nodes, dtype=torch.bool)
test_mask[:len(y)] = True
results_augmented = train_and_evaluate(model_augmented, data_augmented, test_mask=test_mask)

