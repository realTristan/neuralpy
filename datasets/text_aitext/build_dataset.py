import typing, torch, base64, csv

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the file
def file_to_sentences(file: str) -> typing.List[str]:
    text: str = open(file, "r").read().strip()

    # Split the text by sentences (. ! ?)
    def split_sentences(sentences: typing.List[str], text: str):
        def _append(sentences: typing.List[str], res: str):
            return (sentences + [res.strip()], "")
        
        # Iterate through the text
        res: str = ""
        for i in range(len(text)):
            res += text[i]
            
            # If the current index is greater than the length of the text,
            # then append the sentence to the list and reset the result
            if i + 1 >= len(text):
                return _append(sentences, res)[0]
            
            # If the next two characters are equal to the end of a sentence, 
            # then append the sentence to the list and reset the result
            match text[i:i+2]:
                case ". " | "? " | "! ":
                    sentences, res = _append(sentences, res)
        
        # Return the sentences
        return sentences

    # Convert the sentence into base64
    def base64_encode(s: str) -> str:
        return str(base64.b64encode(bytes(s, "utf-8")), "utf-8")
    
    # Remove empty sentences
    sentences = split_sentences([], text)
    return [base64_encode(s) for s in sentences if s != ""]


# Write the sentences to the csv file
def write_to_csv(file: str) -> None:
    with open(file, "w") as output:
        writer = csv.writer(output)

        # Create a new column for the images and labels
        writer.writerow(["text", "label"])

        # Read all the images from the text.txt file
        text: typing.List[str] = file_to_sentences("texts/text.txt")
        for sentence in text:
            writer.writerow([sentence, 1])  # 1 for not ai made
            
        # Read all the images from the text_ai.txt file
        text_ai: typing.List[str] = file_to_sentences("texts/text_ai.txt")
        for sentence in text_ai:
            writer.writerow([sentence, 0])  # 0 for ai made
            
            
def read_csv(file: str) -> typing.List[tuple[str, int]]:
    with open(file, "r") as output:
        reader = csv.reader(output)
        next(reader)  # Skip the first row

        # Iterate through the rows
        res: typing.List[tuple[str, int]] = []
        for row in reader:
            res.append((row[0], int(row[1])))
        return res


def to_tensor(sentence: str) -> torch.Tensor:
    # Base64 decode
    def base64_decode(s: str) -> str:
        return str(base64.b64decode(s), "utf-8")
    sentence = base64_decode(sentence)
    
    # Remove double spaces
    def remove_double_spaces(s: str) -> str:
        while "  " in s:
            s = s.replace("  ", " ")
        return s

    # Convert the sentence into a list of numbers
    sentence = remove_double_spaces(sentence)
    
    # Convert the sentence into a tensor
    return torch.ByteTensor(list(sentence.encode("utf-8"))).to(device) #torch.tensor(sentence)


def all_to_tensor(data: typing.List[tuple[str, int]]) -> typing.List[tuple[torch.Tensor, torch.Tensor]]:
    new_data: typing.List[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(len(data)):
        label_tensor: torch.Tensor = torch.tensor(data[i][1]).to(device)
        text_tensor: torch.Tensor = to_tensor(data[i][0])
        new_data.append((text_tensor, label_tensor))
    return new_data


def padding(all_tensors: typing.List[tuple[torch.Tensor, torch.Tensor]]):
    max_len: int = 0
    for tensor in all_tensors:
        if len(tensor[0]) > max_len:
            max_len = len(tensor[0])
    
    for i in range(len(all_tensors)):
        while len(all_tensors[i][0]) < max_len:
            tmp: typing.List[tensor.Tensor] = list(all_tensors[i][0])
            tmp = torch.cat((all_tensors[i][0], torch.tensor([0]).to(device)))
            all_tensors[i] = (tmp, all_tensors[i][1])
    return all_tensors


if __name__ == "__main__":
    write_to_csv("text_aitext.csv")
    csv_data: typing.List[tuple[str, int]] = read_csv("text_aitext.csv")
    all_tensors: typing.List[tuple[torch.Tensor, torch.Tensor]] = all_to_tensor(csv_data)
    padded: typing.List[tuple[torch.Tensor, torch.Tensor]] = padding(all_tensors)
    print(padded)
