import csv, os

with open("datasets/cats_dogs/dogs_cats.csv", "r") as input:
    with open("datasets/cats_dogs/dogs_cats.csv", "w") as output:
        reader = csv.reader(input)
        writer = csv.writer(output)

        # Create a new column for the images and labels
        writer.writerow(["image", "label"])

        # Read all the images from the images/dogs folder
        images = os.listdir("datasets/cats_dogs/images/dogs")
        for image in images:
            writer.writerow([f"dogs/{image}", 1])  # 1 for dogs

        # Read all the images from the images/cats folder
        images = os.listdir("datasets/cats_dogs/images/cats")
        for image in images:
            writer.writerow([f"cats/{image}", 0])
