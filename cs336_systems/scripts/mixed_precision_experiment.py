import torch
import torch.nn as nn
import torch.nn.functional as F

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)



s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.relu(self.fc1(x))
    #     x = self.ln(x)
    #     x = self.fc2(x)
    #     return x
    
    def forward(self, x):
        print("input dtype:", x.dtype)

        x = self.fc1(x)
        print("fc1 output dtype:", x.dtype)

        x = self.relu(x)
        print("relu output dtype:", x.dtype)

        x = self.ln(x)
        print("layer norm output dtype:", x.dtype)

        x = self.fc2(x)
        print("logits dtype:", x.dtype)

        return x


device = "cuda"
model = ToyModel(in_features=16, out_features=8).to(device)

print("parameter dtype before autocast:")
for name, param in model.named_parameters():
    print(name, param.dtype)

print("parameter dtype after autocast:")
x = torch.randn(4, 16, device=device)
target = torch.randint(0, 8, (4,), device=device)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    print("loss dtype:", loss.dtype)

loss.backward()

print("\ngradient dtype:")
for name, param in model.named_parameters():
    print(name, param.grad.dtype)

