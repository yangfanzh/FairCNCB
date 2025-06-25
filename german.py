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


data = pd.read_csv('German.csv')

sensitive_attr = 'Gender'
target_attr = 'GoodCustomer'

features = data.drop(columns=[target_attr])
labels = data[target_attr]

categorical_cols = ['PurposeOfLoan']
numeric_cols = [col for col in features.columns if col not in categorical_cols + [sensitive_attr]]

encoder = OneHotEncoder(drop='first', sparse=False)
categorical_features = encoder.fit_transform(features[categorical_cols])

scaler = StandardScaler()
numeric_features = scaler.fit_transform(features[numeric_cols])

processed_features = np.hstack([numeric_features, categorical_features])

sensitive_features = (features[sensitive_attr] == 'Male').astype(int).values.reshape(-1, 1)

X = torch.FloatTensor(processed_features)
sensitive = torch.FloatTensor(sensitive_features)
y = torch.LongTensor((labels.values + 1) // 2)


X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

from sklearn.neighbors import kneighbors_graph


adj_matrix = kneighbors_graph(processed_features, n_neighbors=5, mode='connectivity', include_self=True)


adj_coo = adj_matrix.tocoo()
edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)


data = Data(x=X, edge_index=edge_index, y=y, sensitive=sensitive)
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[:len(X_train)] = True
test_mask[len(X_train):] = True
data.train_mask = train_mask
data.test_mask = test_mask



class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, sensitive_dim=1):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + sensitive_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)

    def forward(self, z, sensitive):
        z = torch.cat([z, sensitive], dim=1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, sensitive_dim=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + sensitive_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, x, sensitive):
        x = torch.cat([x, sensitive], dim=1)
        return self.net(x)



input_dim = processed_features.shape[1]
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim, input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)


g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)


criterion = nn.BCELoss()


def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion,
              data_loader, num_epochs=1000, latent_dim=100):
    for epoch in range(num_epochs):
        for real_data, real_sensitive in data_loader:
            batch_size = real_data.size(0)

            real_data = real_data.to(device)
            real_sensitive = real_sensitive.to(device)

            z = torch.randn(batch_size, latent_dim).to(device)

            counter_sensitive = 1 - real_sensitive
            fake_data = generator(z, counter_sensitive)


            d_optimizer.zero_grad()

            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_data, real_sensitive)
            d_loss_real = criterion(real_output, real_labels)

            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_data.detach(), counter_sensitive)
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()


            g_optimizer.zero_grad()


            output = discriminator(fake_data, counter_sensitive)
            g_loss = criterion(output, real_labels)

            g_loss.backward()
            g_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')



from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train, sensitive_train)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion,
          data_loader, num_epochs=1000, latent_dim=latent_dim)



def generate_counterfactuals(generator, num_samples, latent_dim=100):
    z = torch.randn(num_samples, latent_dim).to(device)

    sensitive_0 = torch.zeros(num_samples, 1).to(device)
    sensitive_1 = torch.ones(num_samples, 1).to(device)


    counter_0_to_1 = generator(z, sensitive_1)
    counter_1_to_0 = generator(z, sensitive_0)

    return counter_0_to_1, counter_1_to_0



num_samples = X_train.shape[0]
counter_0_to_1, counter_1_to_0 = generate_counterfactuals(generator, num_samples)

counter_0_to_1 = counter_0_to_1.cpu().detach().numpy()
counter_1_to_0 = counter_1_to_0.cpu().detach().numpy()



def augment_graph(original_features, original_sensitive, counter_0_to_1, counter_1_to_0):

    original_sensitive = original_sensitive.flatten()
    idx_0 = np.where(original_sensitive == 0)[0]
    idx_1 = np.where(original_sensitive == 1)[0]


    selected_idx_0 = np.random.choice(idx_0, len(counter_0_to_1), replace=False)
    selected_idx_1 = np.random.choice(idx_1, len(counter_1_to_0), replace=False)


    counter_sensitive_0_to_1 = np.ones((len(counter_0_to_1), 1))
    counter_sensitive_1_to_0 = np.zeros((len(counter_1_to_0), 1))


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

    augmented_labels = np.concatenate([
        y.numpy(),
        y.numpy()[selected_idx_0],
        y.numpy()[selected_idx_1]
    ])

    return augmented_features, augmented_sensitive, augmented_labels



augmented_features, augmented_sensitive, augmented_labels = augment_graph(
    processed_features, sensitive.numpy(), counter_0_to_1, counter_1_to_0
)


augmented_adj = kneighbors_graph(augmented_features, n_neighbors=5, mode='connectivity', include_self=True)
augmented_coo = augmented_adj.tocoo()
augmented_edge_index = torch.tensor([augmented_coo.row, augmented_coo.col], dtype=torch.long)

augmented_data = Data(
    x=torch.FloatTensor(augmented_features),
    edge_index=augmented_edge_index,
    y=torch.LongTensor(augmented_labels),
    sensitive=torch.FloatTensor(augmented_sensitive)
)


train_size = len(X_train)
test_size = len(X_test)
augmented_train_mask = torch.zeros(augmented_data.num_nodes, dtype=torch.bool)
augmented_test_mask = torch.zeros(augmented_data.num_nodes, dtype=torch.bool)

augmented_train_mask[:train_size + 2 * num_samples] = True
augmented_test_mask[train_size:train_size + test_size] = True

augmented_data.train_mask = augmented_train_mask
augmented_data.test_mask = augmented_test_mask



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

input_dim = augmented_data.num_features
hidden_dim = 64
output_dim = 2

model = GCN(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y.to(device)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        acc = accuracy_score(data.y[data.test_mask].numpy(), pred[data.test_mask].cpu().numpy())

        test_indices = torch.where(data.test_mask)[0].numpy()
        sensitive_test = data.sensitive[test_indices].numpy().flatten()
        y_test = data.y[test_indices].numpy()
        pred_test = pred[test_indices].cpu().numpy()
        mask_0 = sensitive_test == 0
        acc_0 = accuracy_score(y_test[mask_0], pred_test[mask_0]) if sum(mask_0) > 0 else 0
        mask_1 = sensitive_test == 1
        acc_1 = accuracy_score(y_test[mask_1], pred_test[mask_1]) if sum(mask_1) > 0 else 0

        fair_metric = abs(acc_0 - acc_1)

    return acc, fair_metric
for epoch in range(200):
    loss = train(model, augmented_data)
    if epoch % 20 == 0:
        acc, fair_metric = test(model, augmented_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Fairness: {fair_metric:.4f}')



def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)


        test_indices = torch.where(data.test_mask)[0].numpy()

        y_true = data.y[test_indices].numpy()
        y_pred = pred[test_indices].cpu().numpy()

        sensitive = data.sensitive[test_indices].numpy().flatten()


        overall_acc = accuracy_score(y_true, y_pred)
        overall_f1 = f1_score(y_true, y_pred)

        results = {}
        for s in [0, 1]:
            mask = sensitive == s
            if sum(mask) == 0:
                continue

            y_true_s = y_true[mask]
            y_pred_s = y_pred[mask]

            results[f'acc_s{s}'] = accuracy_score(y_true_s, y_pred_s)
            results[f'f1_s{s}'] = f1_score(y_true_s, y_pred_s)
            results[f'tpr_s{s}'] = np.sum((y_true_s == 1) & (y_pred_s == 1)) / np.sum(y_true_s == 1)
            results[f'fpr_s{s}'] = np.sum((y_true_s == 0) & (y_pred_s == 1)) / np.sum(y_true_s == 0)

        # 计算公平性指标
        if 'acc_s0' in results and 'acc_s1' in results:
            results['acc_diff'] = abs(results['acc_s0'] - results['acc_s1'])
        if 'tpr_s0' in results and 'tpr_s1' in results:
            results['tpr_diff'] = abs(results['tpr_s0'] - results['tpr_s1'])
        if 'fpr_s0' in results and 'fpr_s1' in results:
            results['fpr_diff'] = abs(results['fpr_s0'] - results['fpr_s1'])

        results['overall_acc'] = overall_acc
        results['overall_f1'] = overall_f1

        return results

fairness_results = evaluate_model(model, augmented_data)
print("Fairness Evaluation Results:")
for k, v in fairness_results.items():
    print(f"{k}: {v:.4f}")


original_model = GCN(input_dim, hidden_dim, output_dim).to(device)
original_optimizer = optim.Adam(original_model.parameters(), lr=0.01, weight_decay=5e-4)


original_data = Data(
    x=X,
    edge_index=edge_index,
    y=y,
    sensitive=sensitive
)


original_train_mask = torch.zeros(original_data.num_nodes, dtype=torch.bool)
original_test_mask = torch.zeros(original_data.num_nodes, dtype=torch.bool)
original_train_mask[:len(X_train)] = True
original_test_mask[len(X_train):] = True
original_data.train_mask = original_train_mask
original_data.test_mask = original_test_mask

# 训练原始模型
for epoch in range(200):
    loss = train(original_model, original_data)
    if epoch % 20 == 0:
        acc, fair_metric = test(original_model, original_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Fairness: {fair_metric:.4f}')


original_results = evaluate_model(original_model, original_data)

