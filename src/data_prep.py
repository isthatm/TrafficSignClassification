import utils
import torch
import torchvision
import matplotlib.pyplot as plt

test_images, test_labels  = utils.pickle_data(file = './Data/pickled_data/primary32_test_dataset')
val_images, val_labels  = utils.pickle_data(file = './Data/pickled_data/primary32_val_dataset')
grayScale_trans = torchvision.transforms.Grayscale()

gray_test_images = grayScale_trans(torch.tensor(test_images).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
utils.pickle_data(file = "./Data/pickled_data/gray32_test_dataset", writeColumns = [gray_test_images, test_labels])
gray_val_images = grayScale_trans(torch.tensor(val_images).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
utils.pickle_data(file = "./Data/pickled_data/gray32_val_dataset", writeColumns = [gray_val_images, val_labels])


# print(type(train_images))
# img = gray_train_images.permute(0, 2, 3, 1)
# plt.imshow(img[0, : ,: ,:], cmap='gray')
# plt.show()
# print(gray_train_images.size())