import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageChops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_MEAN = [.485, .456, .406]
_STD = [.229, .224, .225]


def load_image(path):
    img = Image.open(path)
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.title('Base image')
    return img


normalize = transforms.Normalize(
    mean=_MEAN,
    std=_STD
)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


def deprocess(image):
    return image * torch.Tensor(_STD).to(device) + torch.Tensor(_MEAN).to(device)


vgg = models.vgg16(pretrained=True).to(device)

module_list = list(vgg.features.modules())


def dreamify(image, layer, iterations, lr):
    x = Variable(preprocess(image).unsqueeze(0).to(device), requires_grad=True)

    # Zero out gradient
    vgg.zero_grad()

    for i in range(iterations):
        out = x
        for j in range(layer):
            out = module_list[j + 1](out)
        loss = out.norm()  # Normalize out
        loss.backward()  # BackPropagation
        x.data = x.data + lr * x.grad.data

    x = x.data.squeeze()
    x.transpose_(0, 1)
    x.transpose_(1, 2)

    # Deprocess the image and make sure it is within the range of 0 and 1. Hint: use deprocess() and np.clip()
    x = np.clip(deprocess(x), 0, 1)
    im = Image.fromarray(np.uint8(x * 255))
    return im


def deep_dream_vgg(image, layer, iterations, lr, octave_scale, num_octaves):
    if num_octaves < 0:
        image_1 = image.filter(ImageFilter.GaussianBlur(2))
        if (image_1.size[0] / octave_scale < 1) or (image_1.size[1] / octave_scale < 1):
            size = image_1.size
        else:
            size = (int(image_1.size[0] / octave_scale), int(image_1.size[1] / octave_scale))

        image_1 = image_1.resize(size.Image.ANTIALIAS)

        image_1 = deep_dream_vgg((image_1, layer, iterations, lr, octave_scale, num_octaves - 1))

        size = image.size

        image_1 = image_1.resize(size, Image.ANTIALIAS)

        image = ImageChops.blend(image, image_1, .6)

    return dreamify(image, layer, iterations, lr).resize(image.size)


sather_img = load_image('data/sather.png')

sather_dream = deep_dream_vgg(sather_img, 5, 5, .3, 2, 20)

sather_dream.save('data/sather_dream.png')