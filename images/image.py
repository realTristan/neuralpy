import cv2

# Convert the image to grayscale
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
def resize(image, size=(28,28)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Run the program
if __name__ == "__main__":
    image_path = "images/8.jpg"
    image = cv2.imread(image_path)
    image = to_gray(image)
    image = resize(image)
    cv2.imwrite(image_path, image)

