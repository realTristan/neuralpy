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
    def train(
        self,
        data: Data,
        clf: Classifier,
        opt: Adam,
        loss_fn: nn.CrossEntropyLoss,
        device: torch.device,
        epochs: int,
    ):
        for epoch in range(epochs):
            def run(epoch: int):
                for batch, (images, labels) in enumerate(data.model):
                    # Acquire the lock
                    self.lock.acquire()

                    # Get the data
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    preds = clf(images)
                    loss = loss_fn(preds, labels)

                    # Backward pass
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # Print the loss
                    print(f"Epoch {epoch} Batch {batch} Loss {loss.item()}")

                    # Release the lock
                    self.lock.release()
            
            # Run the thread
            thread = threading.Thread(target=run, args=(epoch,))
            thread.start()
            thread.join()
