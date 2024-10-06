import utils
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

# # 1. How to load pickled data
# train_images, train_labels  = utils.pickle_data(file = './Data/pickled_data/primary32_train_dataset')
# test_images, test_labels    = utils.pickle_data(file = './Data/pickled_data/primary32_test_dataset')
# val_images, val_labels      = utils.pickle_data(file = './Data/pickled_data/primary32_val_dataset')

# print(type(train_images), type(test_images))

# # 2. How to pickle data
# grayScale_trans = torchvision.transforms.Grayscale()
# gray_train_images = grayScale_trans(train_images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
# utils.pickle_data(file = "./Data/pickled_data/gray32_train_dataset", writeColumns = [gray_train_images, train_labels])
# gray_test_images = grayScale_trans(test_images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
# utils.pickle_data(file = "./Data/pickled_data/gray32_test_dataset", writeColumns = [gray_test_images, test_labels])
# gray_val_images = grayScale_trans(val_images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
# utils.pickle_data(file = "./Data/pickled_data/gray32_val_dataset", writeColumns = [gray_val_images, val_labels])

# print(type(train_images))
# img = gray_train_images.permute(0, 2, 3, 1)
# plt.imshow(img[0, : ,: ,:], cmap='gray')
# plt.show()
# print(gray_train_images.size())

def crop_image_from_annotatation():
    # 3. Crop object from yolo-annotated files
    DATA_ROOT_DIR = r"C:\Users\Admin\Documents\6. Data\VietnamSigns"
    IMGS_DIR      = os.path.join(DATA_ROOT_DIR, "images")
    LABELS_DIR    = os.path.join(DATA_ROOT_DIR, "labels")
    OUTPUT_DIR    = os.path.join(DATA_ROOT_DIR, "class_images")

    # Loop through the images
    for filename in os.listdir(IMGS_DIR):
        # Get the full path
        img_filepath = os.path.join(IMGS_DIR, filename)

        # Get the corresponding annotation file path
        base_name, _ = os.path.splitext(filename)
        annotation_filename = base_name + '.txt'
        annotation_path = os.path.join(LABELS_DIR, annotation_filename)

        # Check if the file is an image (you can adjust the extensions as needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            utils.crop_and_save_image(img_filepath, annotation_path, base_name, OUTPUT_DIR)

def augment_data_flow():
    ROOT_INPUT_DIR  = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\class_images"
    ROOT_OUTPUT_DIR = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\augmented_class"
    TARGET_NUM_DATA = 250
    preprocessor = utils.PreProcessing()
    class_dirs = [d for d in os.listdir(ROOT_INPUT_DIR) if os.path.isdir(os.path.join(ROOT_INPUT_DIR, d))]
    for input_dir_name in class_dirs:
        input_dir_path = os.path.join(ROOT_INPUT_DIR, input_dir_name)
        augmented_dir  = os.path.join(ROOT_OUTPUT_DIR, input_dir_name)
        preprocessor.augment_data(input_dir_path, augmented_dir, TARGET_NUM_DATA)

def preprocess_data():
    ROOT_INPUT_DIR   = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\augmented_class"
    #ROOT_INPUT_DIR  = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\test_preprocessing_input"
    ROOT_OUTPUT_DIR = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\preprocess_classes"
    #ROOT_OUTPUT_DIR = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\test_pre_processing_output"
    preprocessor = utils.PreProcessing()
    class_dirs = [d for d in os.listdir(ROOT_INPUT_DIR) if os.path.isdir(os.path.join(ROOT_INPUT_DIR, d))]
    for input_dir_name in class_dirs:
        input_dir_path   = os.path.join(ROOT_INPUT_DIR, input_dir_name)
        preprocessed_dir = os.path.join(ROOT_OUTPUT_DIR, input_dir_name)
        preprocessor.apply_preprocessing(input_dir_path, preprocessed_dir)

"""
    Load data from a dir with folders whose names are classes labels -> (image tensor, label tensor)
"""
def load_data_to_tensor():
    INPUT_DIR  = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\augmented_class" 
    OUTPUT_DIR = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\pickled_augmented_preprocessed"
    OUTPUT_FILENAME  = "colored_augmented_dataset" 
    OUTPUT_FILE_PATH =  os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    #img_tensor, label_tensor = utils.load_data(INPUT_DIR) # images must be preprocessed/have the same size
    #utils.pickle_data(file = OUTPUT_FILE_PATH, writeColumns = [img_tensor, label_tensor]) # Pickle data

    train_images, train_labels  = utils.pickle_data(file = OUTPUT_FILE_PATH) # Read pickled data

    # Data Loader
    portion    = 0.8
    train_size = int(portion * len(train_images))
    val_size   = len(train_images) - train_size
    train_dataset, val_dataset = random_split(list(zip(train_images, train_labels)), [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    SAVE_DIR = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\test_pre_processing_output"
    for i, (input, target) in enumerate(train_loader):
        #print(input.size())
        processed_batch = preprocess_batch(input)
        save_batch(processed_batch, SAVE_DIR)
        print("SAVING results to {SAVE_DIR}}")
        break
        

    # display_img  = train_images[0]
    # fomrmat_to_display = display_img.permute(1, 2, 0)
    # plt.imshow(fomrmat_to_display,cmap='gray')
    # plt.show()
    print(train_images.size())
    print(train_labels.size())

@staticmethod
# Brief: Save images within a batch [batch_size, C, H, W] into a directory
def save_batch(batch, save_dir: str):
    to_pil = transforms.ToPILImage()
    os.makedirs(save_dir, exist_ok=True)
    for i, img_tensor in enumerate(batch):
        img = to_pil(img_tensor[0])  # Convert tensor to PIL image
        img.save(os.path.join(save_dir, f'image_{i}.png'))  # Save image with a unique name

def modify_parameters(line):
    parts = line.split()
    if len(parts) == 5:
        # Modify a random parameter (index 1 to 4) with a random value
        parts[0] = "0"
    return " ".join(parts)

def preprocess_batch(batch_tensor):
      preprocessed_image = []
      idx = 0
      for img_tensor in batch_tensor:
         # Convert tensor to numpy array
        #print("{}: {}".format(idx, img_tensor.size()))
        img_np = img_tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

         # Convert from [0, 1] or [0, 255] to [0, 255] and ensure uint8 type
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        img_res = preprocess(img_np)
        img_tensor = torch.from_numpy(img_res).unsqueeze(0)
        preprocessed_image.append(img_tensor)
        idx += 1

      preprocessed_tensor = torch.stack(preprocessed_image)
      return preprocessed_tensor

def preprocess(image):
    if image.shape[2] > 1:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Noise reduction
      image = cv2.GaussianBlur(image, (3, 3), 1,5)
      # Histogram equalization
      image = cv2.equalizeHist(image)
      # Image eroding
      image = cv2.erode(image, (3, 3))
      # Resize image
      image = cv2.resize(image, (32, 32))
    return image

if __name__ == '__main__':
    # crop_image_from_annotatation()
    # augment_data_flow()
    #preprocess_data()
    load_data_to_tensor()
    pass
