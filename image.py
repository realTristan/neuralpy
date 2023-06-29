from torchvision.transforms import ToTensor
from constants import DEVICE
import torch, PIL, cv2


class Image:
    @staticmethod
    def to_tensor(image, channels: int = 3) -> torch.Tensor:
        # Convert the image to a tensor
        image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0).to(DEVICE)

        # Update channels
        if image_tensor.shape[1] != channels:
            image_tensor = image_tensor.repeat(1, channels, 1, 1)

        # Return the image tensor
        return image_tensor

    @staticmethod
    def format(image_path: str, size: tuple = (300, 300)):
        # Convert the image to grayscale
        def to_gray(image):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image
        def resize(image, size):
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        # Read the image and convert it to grayscale then resize it
        image = cv2.imread(image_path)
        image = to_gray(image)
        image = resize(image, size)

        # Return the image
        return image


# if __name__ == "__main__":
# image = Image.format("images/img.png")
# cv2.imwrite("images/img.png", image)
