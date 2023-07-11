import csv
from text_to_tensor import text_to_sentences

# Write the sentences to the csv file
def write_csv(file: str) -> None:
    with open(file, "w") as output:
        writer = csv.writer(output)

        # Create a new column for the text and labels
        writer.writerow(["text", "label"])

        # Read all the text from the text.txt file
        sentences = text_to_sentences(open("texts/text.txt", "r").read().strip())
        [writer.writerow([s, 1]) for s in sentences]

        # Read all the text from the text_ai.txt file
        sentences = text_to_sentences(open("texts/text_ai.txt", "r").read().strip())
        [writer.writerow([s, 0]) for s in sentences]

# Read the csv file
def read_csv(file: str):
    with open(file, "r") as output:
        reader = csv.reader(output)
        next(reader)  # Skip the first row
        return [(row[0], int(row[1])) for row in reader if len(row) == 2]