from model import Model
import threading, torch


class Trainer:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    # Train the model
    def train(
        self,
        data: torch.utils.data.DataLoader,
        model: Model,
        opt: torch.optim.Adam,
        loss_fn: torch.nn.CrossEntropyLoss,
        device: torch.device,
        epochs: int,
        channels: int = 3,
    ) -> None:
        for epoch in range(epochs):
            # Function for running in a thread
            def run(epoch: int):
                for batch, (images, labels) in enumerate(data):
                    # Acquire the lock
                    self.lock.acquire()

                    # Get the data
                    images, labels = images.to(device), labels.to(device)
                    
                    # Update channels if needed
                    if images.shape[1] != channels:
                        # Create empty values (of 1) so that the number of channels is 
                        # equal to the number of channels in the model
                        images = images.repeat(1, channels, 1, 1)

                    # Forward pass
                    preds = model(images) # Get the predictions
                    loss = loss_fn(preds, labels) # Calculate the loss (prediction, actual) (pred - act) ** 2

                    # Backward pass
                    opt.zero_grad() # Revert back to the original gradient
                    loss.backward() # Perform the back propagation
                    opt.step() # Update the weights

                    # Print the loss
                    print(f"Epoch {epoch} Batch {batch} Loss {loss.item()}")

                    # Release the lock
                    self.lock.release()

            # Run the thread
            thread = threading.Thread(target=run, args=(epoch,))
            thread.start()
            thread.join()
