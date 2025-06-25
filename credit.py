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


np.random.seed(42)
torch.manual_seed(42)

data = pd.read_csv('credit.csv')

sensitive_attr = 'Married'
target_attr = 'NoDefaultNextMonth'

features = data.drop(columns=[target_attr])
target = data[target_attr]

continuous_features = ['Age', 'MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months',
                       'MonthsWithZeroBalanceOverLast6Months', 'MonthsWithLowSpendingOverLast6Months',
                       'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount',
                       'MostRecentPaymentAmount', 'TotalOverdueCounts', 'TotalMonthsOverdue']

categorical_features = ['EducationLevel', 'HistoryOfOverduePayments']


scaler = StandardScaler()
features[continuous_features] = scaler.fit_transform(features[continuous_features])


features = pd.get_dummies(features, columns=categorical_features)


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)


X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)



class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


input_dim = X_train.shape[1]
latent_dim = 100
batch_size = 64
epochs = 500
lr = 0.0002

generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

criterion = nn.BCELoss()

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, (real_data, _) in enumerate(dataloader):
        batch_size = real_data.size(0)


        optimizer_D.zero_grad()


        real_labels = torch.ones(batch_size, 1)
        output = discriminator(real_data)
        d_loss_real = criterion(output, real_labels)


        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        output = discriminator(fake_data.detach())
        d_loss_fake = criterion(output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()

        output = discriminator(fake_data)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')



def generate_counterfactuals(generator, original_data, sensitive_idx, target_value, num_samples):

    noise = torch.randn(num_samples, latent_dim)


    synthetic_samples = generator(noise).detach()

    synthetic_samples[:, sensitive_idx] = target_value

    synthetic_samples = torch.clamp(synthetic_samples, -1, 1)

    return synthetic_samples


sensitive_idx = features.columns.get_loc(sensitive_attr)


num_counterfactuals = 500

counterfactuals_married = generate_counterfactuals(generator, X_train_tensor, sensitive_idx, 1, num_counterfactuals)

counterfactuals_unmarried = generate_counterfactuals(generator, X_train_tensor, sensitive_idx, 0, num_counterfactuals)

X_train_extended = torch.cat([X_train_tensor, counterfactuals_married, counterfactuals_unmarried], dim=0)

y_counterfactuals = torch.cat([
    torch.ones(num_counterfactuals, 1),
    torch.zeros(num_counterfactuals, 1)
])

y_train_extended = torch.cat([y_train_tensor, y_counterfactuals], dim=0)


def build_graph(features, k=5):

    features_np = features.numpy()

    distances = np.zeros((features_np.shape[0], features_np.shape[0]))
    for i in range(features_np.shape[0]):
        for j in range(features_np.shape[0]):
            distances[i, j] = np.linalg.norm(features_np[i] - features_np[j])

    adj = np.zeros_like(distances)
    for i in range(distances.shape[0]):
        idx = np.argpartition(distances[i], k)[:k + 1]
        adj[i, idx] = 1
        adj[idx, i] = 1

    np.fill_diagonal(adj, 0)

    return adj

adj_matrix = build_graph(X_train_extended)

edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long)

graph_data = Data(
    x=X_train_extended,
    edge_index=edge_index,
    y=y_train_extended
)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return self.sigmoid(x)


input_dim = X_train_extended.shape[1]
hidden_dim = 64
output_dim = 1

model = GCN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(graph_data.x, graph_data.edge_index)
    loss = criterion(outputs, graph_data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

def evaluate(model, X, y, sensitive_attr, sensitive_idx):
    model.eval()
    with torch.no_grad():

        test_adj = build_graph(X, k=5)
        test_edge_index = torch.tensor(np.array(np.where(test_adj)), dtype=torch.long)
        test_data = Data(x=X, edge_index=test_edge_index)

        outputs = model(test_data.x, test_data.edge_index)
        predictions = (outputs > 0.5).float()
        y_np = y.numpy().flatten()
        pred_np = predictions.numpy().flatten()
        prob_np = outputs.numpy().flatten()
        sensitive_np = X[:, sensitive_idx].numpy().flatten()

        acc = accuracy_score(y_np, pred_np)
        f1 = f1_score(y_np, pred_np)
        auc = roc_auc_score(y_np, prob_np)

        sp0 = np.mean(pred_np[sensitive_np == 0]) - np.mean(pred_np)
        sp1 = np.mean(pred_np[sensitive_np == 1]) - np.mean(pred_np)
        sp = max(abs(sp0), abs(sp1))

        y1_idx = y_np == 1
        eo0 = np.mean(pred_np[(sensitive_np == 0) & y1_idx]) - np.mean(pred_np[y1_idx])
        eo1 = np.mean(pred_np[(sensitive_np == 1) & y1_idx]) - np.mean(pred_np[y1_idx])
        eo = max(abs(eo0), abs(eo1))

        return {
            'Accuracy': acc,
            'F1': f1,
            'AUC': auc,
            'Statistical Parity': sp,
            'Equal Opportunity': eo
        }



test_results = evaluate(model, X_test_tensor, y_test_tensor, sensitive_attr, sensitive_idx)


for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

orig_adj_matrix = build_graph(X_train_tensor)
orig_edge_index = torch.tensor(np.array(np.where(orig_adj_matrix)), dtype=torch.long)
orig_graph_data = Data(x=X_train_tensor, edge_index=orig_edge_index, y=y_train_tensor)

model_orig = GCN(input_dim, hidden_dim, output_dim)
optimizer_orig = optim.Adam(model_orig.parameters(), lr=0.01)

for epoch in range(num_epochs):
    model_orig.train()
    optimizer_orig.zero_grad()
    outputs = model_orig(orig_graph_data.x, orig_graph_data.edge_index)
    loss = criterion(outputs, orig_graph_data.y)
    loss.backward()
    optimizer_orig.step()

orig_test_results = evaluate(model_orig, X_test_tensor, y_test_tensor, sensitive_attr, sensitive_idx)

