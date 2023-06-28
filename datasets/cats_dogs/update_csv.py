import csv, os

with open("datasets/cats_dogs/dogs_cats.csv", "w") as output:
    writer = csv.writer(output)

    # Create a new column for the images and labels
    writer.writerow(["image", "label"])

    # Read all the images from the images/dogs folder
    images = os.listdir("datasets/cats_dogs/images/dogs")
    [writer.writerow([f"dogs/{image}", 1]) for image in images]

    # Read all the images from the images/cats folder
    images = os.listdir("datasets/cats_dogs/images/cats")
    [writer.writerow([f"cats/{image}", 0]) for image in images]
