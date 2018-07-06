from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms as transforms

#from util import is_image_file, load_img
import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #img = img.resize((256, 256), Image.BICUBIC)
    return img


# def save_img(image_tensor, filename):
#     image_numpy = image_tensor.float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     image_numpy = image_numpy.astype(np.uint8)
#     image_pil = Image.fromarray(image_numpy)
#     image_pil.save(filename)
#     print("Image saved as {}".format(filename))




class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.photo_path = join(image_dir, "new")
        self.sketch_path = join(image_dir, "frames")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]

        

        self.transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    #tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)