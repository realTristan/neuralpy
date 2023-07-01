import torch
from config import EPOCHS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataset):
    model.train()
    # BinaryCrossEntropyWithLogits
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y = y.unsqueeze(0).unsqueeze(0).float()
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} Loss: {loss.item()}")


def test(model, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y_0 = dataset[i]
            x = x.to(DEVICE)
            y = y_0.to(DEVICE)
            y = y.unsqueeze(0).unsqueeze(0).float()
            output = model(x)
            if output[0][0] > 0.5 and y_0.item() == 1:
                if y[0][0] == 1:
                    correct += 1
            elif output[0][0] < 0.5 and y_0.item() == 0:
                if y[0][0] == 0:
                    correct += 1
    print(f"Accuracy: {correct / (len(dataset) + 1) * 100}%")
