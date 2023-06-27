import torch
from torch import nn, save
from torch.optim import Adam
from data import Data
from classifier import Classifier
import threading

class Trainer:
  def __init__(self):
    self.lock = threading.Lock()
    
  # Train the model
  def train(self, data: Data, clf: Classifier, opt: Adam, loss_fn: nn.CrossEntropyLoss, device: torch.device, epochs: int):
    for epoch in range(epochs):
      # Acquire the lock
      # self.lock.acquire()
      
      # Run the training loop
      def run(epoch: int):
        for batch, (images, labels) in enumerate(data.model):
          images, labels = images.to(device), labels.to(device)

          # Forward pass
          preds = clf(images)
          loss = loss_fn(preds, labels)
          
          # Backward pass
          opt.zero_grad()
          loss.backward()
          opt.step()
          
          # Print the loss
          if batch % 100 == 0:
            print(f"Epoch {epoch} Batch {batch} Loss {loss.item()}")
      
      run(epoch)      
      # Start the thread
      # threading.Thread(target=run, args=(epoch,)).start()
  
  
  # Save the model to a file
  @staticmethod
  def save(clf: Classifier, path: str):
    with open(path, "wb") as f:
      save(clf.state_dict(), f)
    