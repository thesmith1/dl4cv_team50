import sys
import torch
import numpy as np
from torch.autograd import Variable 
from torchvision import transforms

base_folder = "/home/paulstpr/dl4cv_team50/"
sys.path.append(base_folder)
sys.path.append("/home/paulstpr/dl4cv_team50/")

from preprocessing.inaturalist_dataset import INaturalistDataset


model_file = base_folder + 'single_network/models/' + 'num-epochs=10_model=resnet50_reg=0_train-conv=True_batch-size=400_lr=0.001_num-fc-layers=1_loss=CrossEntropyLoss_optimizer=Adam.pth'
size = 224
batch_size = 1
transform = transforms.ToTensor()
use_supercategory = True

def new_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    # x = self.fc(x)
    return x

data_set = INaturalistDataset(base_folder + "data_preprocessed_" + str(size) + "/", base_folder + "annotations/train2017_vis.json", transform, False)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

model = torch.load(open(model_file, "rb"))
model.cuda = False
model.forward = new_forward

N = len(data_loader)*batch_size

X = np.zeros((N, 2048))
y = np.zeros((N,))
for i, (data, targets) in enumerate(data_loader):

    target1, target2 = targets
    data = Variable(data)
    data = data.cuda()
    output = model.forward(model, data)

    result = output.data.cpu().numpy()
    X[i] = result
    y[i] = target1 if use_supercategory else target2


print(X, X.shape)
np.save("images.npy", X)
np.save("labels.npy", y)
print("done.")
