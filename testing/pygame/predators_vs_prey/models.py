import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prey model
class PreyModel(torch.nn.Module):
    def __init__(self):
        super(PreyModel, self).__init__()
        self.lstm1 = torch.nn.LSTM(5, 64)
        self.linear1 = torch.nn.Linear(64, 128)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.lstm2 = torch.nn.LSTM(128, 128)
        self.linear2 = torch.nn.Linear(128, 1)
        self.sigmoid2 = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x, _ = self.lstm2(x)
        x = self.linear2(x)
        return self.sigmoid2(x)
        
    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def _train(self, data, result):
        self.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        x, y = data, result
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y = y.unsqueeze(0)
        output = self(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f"Loss: {loss.item()}")


# Predator model
class PredatorModel(torch.nn.Module):
    def __init__(self):
        super(PredatorModel, self).__init__()
        self.lstm1 = torch.nn.LSTM(3, 64)
        self.linear1 = torch.nn.Linear(64, 128)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.lstm2 = torch.nn.LSTM(128, 128)
        self.linear2 = torch.nn.Linear(128, 1)
        self.sigmoid2 = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x, _ = self.lstm2(x)
        x = self.linear2(x)
        return self.sigmoid2(x)
        
    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def _train(self, data, result):
        self.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        x, y = data, result
        if x is None:
            return
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y = y.unsqueeze(0)
        output = self(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f"Loss: {loss.item()}")
     
     
     
 
# Models
PREY_MODEL: PreyModel = PreyModel()
PREDATOR_MODEL: PredatorModel = PredatorModel()