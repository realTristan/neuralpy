from models import ImageModel
import threading, torch


class Trainer:
    @staticmethod
    def train(
        data: torch.utils.data.DataLoader,
        model: ImageModel,
        opt: torch.optim.Adam,
        loss_fn: torch.nn.CrossEntropyLoss,
        device: torch.device,
        epochs: int,
        channels: int = 3,
    ) -> None:
        # Threading lock
        lock: threading.Lock = threading.Lock()
        
        # Train the model
        for epoch in range(epochs):
            # Function for running in a thread
            def run(epoch: int):
                for batch, (value, label) in enumerate(data):
                    # Acquire the lock
                    lock.acquire()

                    # Get the data
                    value, label = value.to(device), label.to(device)
                    
                    # Update channels if needed
                    if value.shape[1] != channels:
                        # Create empty values (of 1) so that the number of channels is 
                        # equal to the number of channels in the model
                        value = value.repeat(1, channels, 1, 1)

                    # Forward pass
                    preds = model(value) # Get the predictions
                    loss = loss_fn(preds, label) # Calculate the loss (prediction, actual) (pred - act) ** 2

                    # Backward pass
                    opt.zero_grad() # Revert back to the original gradient
                    loss.backward() # Perform the back propagation
                    opt.step() # Update the weights

                    # Print the loss
                    print(f"Epoch {epoch} Batch {batch} Loss {loss.item()}")

                    # Release the lock
                    lock.release()

            # Run the thread
            thread = threading.Thread(target=run, args=(epoch,))
            thread.start()
            thread.join()
