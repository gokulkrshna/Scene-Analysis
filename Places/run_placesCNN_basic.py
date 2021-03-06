# PlacesCNN for scene classification
#
from __future__ import print_function
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import sys

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = 'whole_%s_places365_python36.pth.tar' % arch


#for using the GPU or not
useGPU = 1
if useGPU == 1:
    model = torch.load(model_file)
else:
    model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
PATH_TO_TEST_IMAGES_DIR = '../pi_images'
img_name = os.path.join(PATH_TO_TEST_IMAGES_DIR,"example.jpg")

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

#print('RESULT ON ' + img_name)
# output the prediction into record.txt
recordFile = open("../record.txt", mode="a")
print("The Scene is classsified as either:", file=recordFile)
for i in range(0, 3):
    print(classes[idx[i]], file=recordFile)
recordFile.close()