import csv, os

with open("healthy_unhealthy.csv", "w") as output:
    writer = csv.writer(output)

    # Create a new column for the images and labels
    writer.writerow(["image", "label"])

    # Read all the images from the images/healthy folder
    images = os.listdir("images/healthy")
    [writer.writerow([f"healthy/{image}", 1]) for image in images]

    # Read all the images from the images/unhealthy folder
    images = os.listdir("images/unhealthy")
    [writer.writerow([f"unhealthy/{image}", 0]) for image in images]
