from torchvision.transforms import ToTensor
import torch, PIL



class Image:
    @staticmethod
    def to_tensor(image: PIL.Image, device: torch.device, channels: int = 3) -> torch.Tensor:
        # Convert the image to a tensor
        image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0).to(device)

        # Update channels
        if image_tensor.shape[1] != channels:
            image_tensor = image_tensor.repeat(1, channels, 1, 1)
        
        # Return the image tensor
        return image_tensor