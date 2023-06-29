import torch
from constants import INPUT_SIZE, DEVICE, EPOCHS
from model import Model
from dataset import Dataset
from utils import base64_encode
from text_to_tensor import text_to_tensor, to_tensor, pad
from csv_utils import read_csv, write_csv


def train(model, dataset):
    model.train()
    criterion = torch.nn.BCELoss()
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


# Test a tensor input
def test_tensor(model, tensor):
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(DEVICE)
        tensor = tensor.unsqueeze(0)
        output = model(tensor)
        if output[0][0] > 0.5:
            print("Not ai made")
        else:
            print("Ai made")


if __name__ == "__main__":
    # Building the dataset
    # write_csv("data.csv")
    csv_data = read_csv("data.csv")
    tensors = to_tensor(csv_data)
    padded = pad(tensors)

    # Training and testing
    model = Model().to(DEVICE)
    dataset = Dataset(padded)
    train(model, dataset)
    test(model, dataset)
    model.save("model.pth")

    # Testing real sentence
    # model.load("model.pth")
    s: str = "The first world war was one of the most brutal and remorseless events in history; ‘the global conflict that defined a century’"
    s = base64_encode(s)
    s_tensor = text_to_tensor(s)
    s_tensor = torch.zeros(INPUT_SIZE).to(DEVICE)
    test_tensor(model, s_tensor)
