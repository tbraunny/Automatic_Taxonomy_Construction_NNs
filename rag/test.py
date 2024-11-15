import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset , DataLoader
from glob import glob
from sklearn.model_selection import train_test_split

# new iteration, do not save images as torch tensors
# improve accuracy measures, print images, confusion matrix, false positives, etc.


# load dataset, return images & labels as numpy arrays
def load_dataset():
    print("Loading image data...")
    basepath = 'C:\\Undergraduate\\Year 4\\CS 482\\final_project\\planesnet\\planesnet'
    images , labels = [] , []

    # loop thru each class to find all files with name starting with label_*
    for label in range(2):
        imgs_path = os.path.join(basepath , f"{label}_*")

        # prep image for addition to list
        for file in glob(imgs_path):
            img = cv2.imread(file)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # convert BGR to RGB
            images.append(img)
            labels.append(label)

    # convert to np arrays for easier handling
    images = np.array(images)
    labels = np.array(labels)

    images = torch.tensor(images , dtype=torch.int64).permute(0 , 3 , 1 , 2)
    labels = torch.tensor(labels , dtype=torch.int64)

    print("Dataset loading complete")

    return images , labels

def data_preprocess(X , y):
    X = X / 255 # normalize
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    #X_train , X_test = torch.tensor(X_train) , torch.tensor(X_test)
    #y_train , y_test = torch.tensor(y_train) , torch.tensor(y_test)

    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=True)

    return train_dataloader , test_dataloader

# define model network
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()
        self.conv1 = nn.Conv2d(3 , 20, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(20 , 64 , 3 , 1 , padding=1)
        self.fc1 = nn.Linear(1600 , 128)
        self.fc2 = nn.Linear(128 , 2)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self , x):
        # convolution 1
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x , 2)
        x = self.bn1(x)
        x = self.dropout1(x)

        # convolution 2
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x , 2)
        x = self.bn2(x)

        x = x.view(x.size(0) , -1) # Flatten the tensor

        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x , dim=1)

        return output

# train model
def train_cnn(train_loader , test_loader):
    print("Training model...")
    cnn = CNN()

    # hyperparameters
    num_epochs = 80
    learning_rate = 0.0001
    optimizer = optim.Adam(cnn.parameters() , learning_rate)
    train_loss , test_loss , train_accuracy , test_accuracy = [] , [] , [] , []
    

    for epoch in range(num_epochs):
        cnn.train()
        running_loss , num_correct = 0 , 0

        for (data , target) in train_loader:
          optimizer.zero_grad()
          train_outputs = cnn(data.float())
          loss = F.nll_loss(train_outputs , target)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          pred = train_outputs.argmax(dim=1, keepdim=True)
          num_correct += pred.eq(target.view_as(pred)).sum().item()

        # train accuracy
        train_acc = num_correct / len(train_loader.dataset)
        train_accuracy.append((train_acc , epoch))
        train_loss.append((running_loss , epoch))

        print(f"Epoch {epoch + 1}: Train Loss: {running_loss} \t Train Accuracy: {train_acc}")

        # test accuracy
        cnn.eval()
        running_loss , num_correct = 0 , 0
        with torch.no_grad():
            for (data , target) in test_loader:
                test_outputs = cnn(data.float())
                loss = F.nll_loss(test_outputs , target)
                running_loss += loss.item()
                pred = test_outputs.argmax(dim=1, keepdim=True)
                num_correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = num_correct / len(test_loader.dataset)

        # store accuracies & losses
        test_accuracy.append((test_acc , epoch))
        test_loss.append((running_loss , epoch))
        print(f"\t Test Loss: {running_loss} \t Test Accuracy: {test_acc}\n") # print loss, accuracy

    torch.save({
            'epoch': num_epochs,
            'model_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss[-1],
            }, f'./planesnet_weights.pt')
    
    print("Training complete")

    return train_loss , test_loss , train_accuracy , test_accuracy


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: " , device)
    images , labels = load_dataset()
    train , test = data_preprocess(images , labels)
    train_loss , test_loss , train_accuracy , test_accuracy = train_cnn(train , test)
    train_loss , test_loss , train_accuracy , test_accuracy = np.array(train_loss) , np.array(test_loss) , np.array(train_accuracy) , np.array(test_accuracy)
    print("Accuracy: " , test_accuracy[-1 , 0])

    # loss
    plt.plot(train_loss[: , 1] , train_loss[: , 0] , color='blue' , label='Train Loss')
    plt.plot(test_loss[: , 1] , test_loss[: , 0] , color='red' , label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")
    plt.savefig('loss.png')

    # accuracy
    plt.plot(train_accuracy[: , 1] , train_accuracy[: , 0] , color='blue' , label='Train Accuracy')
    plt.plot(test_accuracy[: , 1] , test_accuracy[: , 0] , color='red' , label='Test Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs vs. Accuracy")
    plt.savefig('accuracy.png')