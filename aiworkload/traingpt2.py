import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
import pyspdk
from collections import defaultdict
from io import BytesIO
import requests
import grpc
import proto.entity_pb2
import proto.entity_pb2_grpc
import sys

TRANSPORT_TYPE = 'TCP'
ADDRESS_FAMILY = 'IPv4'
TARGET_IP = '127.0.0.1'
TARGET_PORT = '4420'
NQN_NAME = 'nqn.2024-02.io.spdk:cnode1'

# The dataset list stores the remote file information in the following format
# A list of lists of the form [[filepath, filesize], [filepath, filesize], ....]
# Populate this during preprocessing
dataset_list = []

# Populate these dictionaries with each files extent information
lbadict = defaultdict(list)
lbacdict = defaultdict(list)


class NVMeFDDataset(Dataset):
    def __init__(self, filelist, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer

        for filename in filelist:
            filepath = filename[0]
            text = pyspdk.fdread(filepath.encode('utf-8'), filename[1]).decode("utf-8")
            tokens = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
            self.examples.append(tokens.input_ids.squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class NVMeKVDataset(Dataset):
    def __init__(self, filelist, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer

        for filename in filelist:
            filepath = filename[0]
            text = pyspdk.kvread(lbadict[filepath], lbacdict[filepath]).decode("utf-8")
            tokens = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
            self.examples.append(tokens.input_ids.squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class HTTPDataset(Dataset):
    def __init__(self, filelist, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer

        for filename in filelist:
            filepath = filename[0]
            text = requests.get('URL/getfile/{}'.format(filepath))
            tokens = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
            self.examples.append(tokens.input_ids.squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class gRPCDataset(Dataset):
    def __init__(self, filelist, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer

        for filename in filelist:
            filepath = filename[0]

            channel = grpc.insecure_channel('localhost:50051')
            stub = proto.entity_pb2.OperateStub(channel)

            request = proto.entity_pb2.FileName(filename=filepath)

            response = stub.GetFile(request)

            text = response.entity.decode("utf-8")
            tokens = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
            self.examples.append(tokens.input_ids.squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class LocalDataset(Dataset):
    def __init__(self, directory, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    tokens = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
                    self.examples.append(tokens.input_ids.squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Collate function to pad batches
def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

def get_dataset(module, tokenizer, block_size):
    if module == 'nvmefd':
        dataset = NVMeFDDataset(dataset_list, tokenizer, block_size)
    elif module == 'nvmekv':
        dataset = NVMeKVDataset(dataset_list, tokenizer, block_size)
    elif module == 'http':
        dataset = HTTPDataset(dataset_list, tokenizer, block_size)
    elif module == 'grpc':
        dataset = gRPCDataset(dataset_list, tokenizer, block_size)
    else:
        dataset = LocalDataset(data_dir, tokenizer, block_size)
    return dataset

if __name__ == '__main__':

    module = sys.argv[1]

    if module == 'nvmefd' or module == 'nvmekv':
        print(pyspdk.spdk_init(TRANSPORT_TYPE.encode('utf-8'), 
                                ADDRESS_FAMILY.encode('utf-8'), 
                                TARGET_IP.encode('utf-8'), 
                                TARGET_PORT.encode('utf-8'), 
                                NQN_NAME.encode('utf-8')))

    # Parameters
    data_dir = "dataset/openwebtext"         # Directory with your text files
    model_name = "gpt2"          # or "gpt2-medium", "gpt2-large"
    block_size = 512
    batch_size = 2
    epochs = 1
    lr = 5e-5

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare dataset and dataloader
    dataset = get_dataset(module, tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")

    print("Training completed and model saved!")
