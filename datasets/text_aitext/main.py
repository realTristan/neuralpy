import typing, torch, base64, csv

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE: int = 152

def base64_encode(s: str) -> str:
    return str(base64.b64encode(s.encode("utf-8")), "utf-8")
    
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

    # Remove empty sentences
    sentences = split_sentences([], text)
    return [base64_encode(s) for s in sentences if s != ""]


# Write the sentences to the csv file
def write_to_csv(file: str) -> None:
    with open(file, "w") as output:
        writer = csv.writer(output)

        # Create a new column for the text and labels
        writer.writerow(["text", "label"])

        # Read all the text from the text.txt file
        text = file_to_sentences("texts/text.txt")
        for sentence in text:
            writer.writerow([sentence, 1])  # 1 for not ai made

        # Read all the text from the text_ai.txt file
        text_ai = file_to_sentences("texts/text_ai.txt")
        for sentence in text_ai:
            writer.writerow([sentence, 0])  # 0 for ai made


def read_csv(file: str):
    with open(file, "r") as output:
        reader = csv.reader(output)
        next(reader)  # Skip the first row

        # Iterate through the rows
        res = []
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
    return torch.ByteTensor(
        list(sentence.encode("utf-8"))).to(DEVICE).float()


def all_to_tensor(data):
    new_data = []
    for i in range(len(data)):
        label_tensor: torch.Tensor = torch.tensor(data[i][1]).to(DEVICE)
        text_tensor: torch.Tensor = to_tensor(data[i][0])
        new_data.append((text_tensor, label_tensor))
    return new_data


def padding(all_tensors):
    max_len: int = 0
    for tensor in all_tensors:
        if len(tensor[0]) > max_len:
            max_len = len(tensor[0])

    for i in range(len(all_tensors)):
        while len(all_tensors[i][0]) < max_len:
            tmp = all_tensors[i][0]
            tmp = torch.cat((all_tensors[i][0], torch.tensor([0]).to(DEVICE)))
            all_tensors[i] = (tmp, all_tensors[i][1])
        unsqueezed = all_tensors[i][0].unsqueeze(0)
        all_tensors[i] = (unsqueezed, all_tensors[i][1])
    return all_tensors


# Create a model and dataset and train the padded data
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(INPUT_SIZE, INPUT_SIZE)
        self.linear = torch.nn.Linear(INPUT_SIZE, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)
    

def train(model, dataset):
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y = y.unsqueeze(0).unsqueeze(0).float()
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} Loss: {loss.item()}")


def test(model, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y_0 = dataset[i]
            x = x.to(DEVICE)
            y = y_0.to(DEVICE)
            y = y.unsqueeze(0).unsqueeze(0).float()
            output = model(x)
            if output[0][0] > 0.5 and y_0.item() == 1:
                if y[0][0] == 1:
                    correct += 1
            elif output[0][0] < 0.5 and y_0.item() == 0:
                if y[0][0] == 0:
                    correct += 1
    print(f"Accuracy: {correct / len(dataset) * 100}%")


# Test a tensor input
def test_tensor(model, tensor):
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(DEVICE)
        tensor = tensor.unsqueeze(0)
        output = model(tensor)
        if output[0][0] > 0.5:
            print("Not ai made")
        else:
            print("Ai made")


if __name__ == "__main__":
    # Building the dataset
    write_to_csv("text_aitext.csv")
    csv_data = read_csv("text_aitext.csv")
    all_tensors = all_to_tensor(csv_data)
    padded = padding(all_tensors)
    
    # Training and testing
    dataset = Dataset(padded)
    model = Model().to(DEVICE)
    train(model, dataset)
    test(model, dataset)
    
    # Testing real sentence
    s: str = "The first world war was one of the most brutal and remorseless events in history; ‘the global conflict that defined a century’"
    s = base64_encode(s)
    s_tensor = to_tensor(s)
    s_tensor = torch.zeros(INPUT_SIZE).to(DEVICE)
    test_tensor(model, s_tensor)
    
