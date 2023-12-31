import csv, os

with open("datasets/drunk_sober/drunk_sober.csv", "w") as output:
    writer = csv.writer(output)

    # Create a new column for the images and labels
    writer.writerow(["image", "label"])

    # Read all the images from the images/drunk folder
    images = os.listdir("datasets/drunk_sober/images/drunk")
    [writer.writerow([f"drunk/{image}", 1]) for image in images]

    # Read all the images from the images/sober folder
    images = os.listdir("datasets/drunk_sober/images/sober")
    [writer.writerow([f"sober/{image}", 0]) for image in images]
