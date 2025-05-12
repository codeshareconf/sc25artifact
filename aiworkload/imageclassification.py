import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# The dataset dictionary stores the remote file information in the following format
# key: Class, which is the folder in which the image files are stored
# value: A list of lists of the form [[filepath, filesize], [filepath, filesize], ....]
# Populate this during preprocessing
dataset_dict = defaultdict(list)

# Populate these dictionaries with each files extent information
lbadict = defaultdict(list)
lbacdict = defaultdict(list)

class NVMeFDImageDataset(Dataset):
    def __init__(self, filedict, transform=None):        
        self.transform = transform
        self.image_paths = []
        self.image_size = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes = sorted(filedict.keys())
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            for img_name in filedict[cls]:
                img_path = os.path.join("/mnt/nvmedrive", img_name[0])
                if img_name[0].lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(img_path)
                    self.image_size.append(img_name[1])
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_bytes = pyspdk.fdread(img_path.encode('utf-8'), self.image_size[idx])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class NVMeKVImageDataset(Dataset):
    def __init__(self, filedict, transform=None):        
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes = sorted(filedict.keys())
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            for img_name in filedict[cls]:
                img_path = os.path.join("/mnt/nvmedrive", img_name[0])
                if img_name[0].lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_bytes = pyspdk.kvread(lbadict[img_path], lbacdict[img_path])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class HTTPImageDataset(Dataset):
    def __init__(self, filedict, transform=None):        
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes = sorted(filedict.keys())
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            for img_name in filedict[cls]:
                img_path =img_name[0]
                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_bytes = requests.get('http://anchor.jf.intel.com:5012/getfile/{}/{}'.format(self.idx_to_class[label],img_path))
        image = Image.open(BytesIO(image_bytes.content)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class gRPCImageDataset(Dataset):
    def __init__(self, filedict, transform=None):        
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes = sorted(filedict.keys())
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            for img_name in filedict[cls]:
                img_path =img_name[0]
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        channel = grpc.insecure_channel('localhost:50051')
        stub = proto.entity_pb2.OperateStub(channel)

        request = proto.entity_pb2.FileName(filename=img_path)

        response = stub.GetFile(request)

        image_bytes = response.entity
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            cls_dir = os.path.join(root_dir, cls)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataset(module, transform):
    if module == 'nvmefd':
        dataset = NVMeFDImageDataset(dataset_dict, transform=transform)
    elif module == 'nvmekv':
        dataset = NVMeKVImageDataset(dataset_dict, transform=transform)
    elif module == 'http':
        dataset = HTTPImageDataset(dataset_dict, transform=transform)
    elif module == 'grpc':
        dataset = gRPCImageDataset(dataset_dict, transform=transform)
    else:
        dataset = LocalImageDataset(root_dir='dataset/imagedata', transform=transform)

    return dataset

if __name__ == '__main__':

    module = sys.argv[1]

    if module == 'nvmefd' or module == 'nvmekv':
        print(pyspdk.spdk_init(TRANSPORT_TYPE.encode('utf-8'), 
                                ADDRESS_FAMILY.encode('utf-8'), 
                                TARGET_IP.encode('utf-8'), 
                                TARGET_PORT.encode('utf-8'), 
                                NQN_NAME.encode('utf-8')))

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = get_dataset(module, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.class_to_idx)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f} Accuracy: {100 * correct/total:.2f}%")

    print("Training Complete.")
