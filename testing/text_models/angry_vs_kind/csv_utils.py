import utils, csv


def read_file_newlines(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        return f.read().splitlines()


def update_csv(csv_path: str, new_rows: list) -> None:
    with open(csv_path, "a") as f:
        for row in new_rows:
            f.write(f"{row[0]}, {row[1]}\n")


def read_angry_texts() -> list[tuple[str, int]]:
    lines: list[str] = read_file_newlines("texts/angry.txt")
    return [(utils.base64_encode(line), 1) for line in lines]


def read_kind_texts() -> list[tuple[str, int]]:
    lines: list[str] = read_file_newlines("texts/kind.txt")
    return [(utils.base64_encode(line), 0) for line in lines]

# Read the csv file
def read_csv(file: str):
    with open(file, "r") as output:
        reader = csv.reader(output)
        next(reader)  # Skip the first row
        return [(row[0], int(row[1])) for row in reader if len(row) == 2]


if __name__ == "__main__":
    update_csv("data.csv", read_angry_texts())
    update_csv("data.csv", read_kind_texts())
