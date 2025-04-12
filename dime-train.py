import torch
import torch.nn.functional as F
from torch_geometric.nn import DimeNet
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from loaders.crystal_dataset import CrystalGraphDataset
from torch.utils.data import random_split

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load QM9 dataset (you can replace this with your own dataset)
path = "data/QM9"
dataset = CrystalGraphDataset("Relaxation_Large")

train_num = int(0.8 * dataset.len)
val_num = int(0.1 * dataset.len)
test_num = dataset.len - train_num - val_num



split = random_split(dataset, [train_num, val_num, test_num])


# Split dataset
train_dataset = split[0]
val_dataset = split[1]
test_dataset = split[2]

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



# Initialize the DimeNet model
model = DimeNet(
    hidden_channels=128,
    out_channels=1,  # Single property prediction
    num_blocks=3,
    num_bilinear=8,
    num_spherical=7,
    num_radial=6,
    envelope_exponent=5,
    cutoff=5.0
).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for pair in train_loader:
        data = pair[0].to(device)  # ✅ Correct way to move PyG Data to GPU
        output = pair[1].to(device)
        
        optimizer.zero_grad()
        breakpoint()
        out = model((data.x[:,0:1]*10).type(torch.int).squeeze(), data.pos, data.batch)  # DimeNet takes atomic numbers, positions, and batch
        breakpoint()
        loss = F.mse_loss(out.squeeze(), data.y[:, target])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y[:, target])
        total_loss += loss.item()
    return total_loss / len(loader)

# Training loop

epochs = 100
for epoch in range(epochs):
    train_loss = train()
    val_loss = evaluate(val_loader)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Final evaluation on test set
test_loss = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}")