import typing, torch
from constants import DEVICE
from utils import base64_encode, base64_decode


# Convert the text into a list of sentences
def text_to_sentences(text: str) -> typing.List[str]:
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
            match text[i : i + 2]:
                case ". " | "? " | "! ":
                    sentences, res = _append(sentences, res)

        # Return the sentences
        return sentences

    # Remove double spaces
    def remove_double_spaces(s: str) -> str:
        while "  " in s:
            s = s.replace("  ", " ")
        return s

    # Convert the sentence into a list of numbers
    text = remove_double_spaces(text)
    text = text.replace("\n", "")

    # Remove empty sentences
    sentences = split_sentences([], text)
    return [base64_encode(s) for s in sentences if s != ""]


# Convert the sentence into a tensor
def text_to_tensor(s: str) -> torch.Tensor:
    s_l = list(base64_decode(s).encode("utf-8"))
    return torch.ByteTensor(s_l).to(DEVICE).float()


# Convert all the data into tensors
def to_tensor(data):
    new_data = []
    for i in range(len(data)):
        label_tensor: torch.Tensor = torch.tensor(data[i][1]).to(DEVICE)
        text_tensor: torch.Tensor = text_to_tensor(data[i][0])
        new_data.append((text_tensor, label_tensor))
    return new_data


# Pad the data
def pad(tensors):
    max_len: int = 0
    for tensor in tensors:
        if len(tensor[0]) > max_len:
            max_len = len(tensor[0])

    for i in range(len(tensors)):
        while len(tensors[i][0]) < max_len:
            tmp = tensors[i][0]
            tmp = torch.cat((tensors[i][0], torch.tensor([0]).to(DEVICE)))
            tensors[i] = (tmp, tensors[i][1])
        unsqueezed = tensors[i][0].unsqueeze(0)
        tensors[i] = (unsqueezed, tensors[i][1])
    return tensors
