import bs4, requests, time, cv2
import numpy as np

dogs_url: str = lambda page: "https://www.bing.com/images/search?q=dog&form=HDRSC4&first=" + str(page)
cats_url: str = lambda page: "https://www.bing.com/images/search?q=cat&form=HDRSC4&first=" + str(page)

# Convert the image to grayscale
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
def resize(image, size=(300,300)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Get the images
def get_images(url: str):
    # Get the HTML
    html: str = requests.get(url).text

    # Parse the HTML
    soup: bs4.BeautifulSoup = bs4.BeautifulSoup(html, "html.parser")

    # Get the images
    images: bs4.element.ResultSet = soup.find_all("img")

    # Return the images
    return images

# Download the images
def download_images(images: bs4.element.ResultSet, path: str):
    # Loop through the images
    for image in images:
        # Get the image source
        src: str = image.get("data-src")
        if src is None or len(src) == 0 or "http" not in src:
            continue
        
        # Download the image
        file_path: str = f"{path}{time.time_ns()}.jpg"
        with open(file_path, "wb") as f:
            content: bytes = requests.get(src).content
            image = cv2.imdecode(np.frombuffer(content, np.uint8), -1)
            image = to_gray(image)
            image = resize(image)
            f.write(cv2.imencode(".jpg", image)[1].tobytes())

# Run the program
if __name__ == "__main__":
    for i in range(1):
        # Download dog images
        url: str = dogs_url(i + 1)
        images: bs4.element.ResultSet = get_images(url)
        download_images(images, "custom_dataset/images/dogs/")
        print(f"Downloaded {len(images)} images from {url}")

        # Download cat images
        url: str = cats_url(i)
        images: bs4.element.ResultSet = get_images(url)
        download_images(images, "custom_dataset/images/cats/")
        print(f"Downloaded {len(images)} images from {url}")
