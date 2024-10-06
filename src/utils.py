import pickle
import random
import os
import torch
import torchvision.transforms as T
import pandas as pd
import shutil
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def label_test(src, csv_file, labeled_test_dir, NoOfCategories):
    """
    This function creates named folders corresponding to 43 categories
    and move the test images to these folders

    `csv_file` and `labeled_test_dir` should have already been in src directly 
    (create a blank folder to store labeled images)

    """
    # Remove the existing folders in the labeled test directory if there is any
    for filename in os.listdir(labeled_test_dir):
        file_path = os.path.join(labeled_test_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    csv_dir = os.path.join(src, csv_file)
    SubTestDir = [os.path.join(labeled_test_dir, str(d)) for d in range(NoOfCategories)]
    
    # Create label folders
    [os.mkdir(test_d) for test_d in SubTestDir]

    testImageDir = pd.read_csv(csv_dir)['Path']
    testImageLabel = pd.read_csv(csv_dir)['ClassId']
    for idx in range(len(testImageLabel)):
        label = testImageLabel[idx]
        shutil.copy(os.path.join(src, testImageDir[idx]), SubTestDir[label])

def pickle_data(file, writeColumns=None):
    """
    Read/Write pickle training/testing data, models to avoid
    loading data again (time consuming)

    :param file: path to pickle file
    :param writeColumns (torch.Tensor or np.ndarray): variables to be saved to pickle file

    :returns :
    If writeColumns = None -> tuple(torch.Tensor)
    tuple()
    """
    if writeColumns is None:
        with open(file, mode="rb") as f:
            dataset = pickle.load(f)
            return tuple( # Convert the pickled data into tensor if it is of any other types
                map(lambda col: torch.tensor(dataset[col], dtype=torch.float32) 
                    if not type(dataset[col]) == torch.Tensor else dataset[col], 
                    ['images', 'labels'])
                )
                # lambda(col) where columns are the inputs
    else:
        with open(file, mode="wb") as f:
            dataset = pickle.dump({"images": writeColumns[0], "labels": writeColumns[1]}, f)
            print("Data is saved in", file)

def load_data(data_dir):
    """Loads a data set and returns two tensors:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    transformer = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    tensor_labels_list = []
    tensor_images_list  = []
    for d in directories:
        # label_dir contains 61 catefories paths
        label_dir = os.path.join(data_dir, d)

        # list subdirectories within each of the 61 categories
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            image = Image.open(f)
            img_tensor   = transformer(image)
            label_tensor = torch.tensor(int(d))
            # images.append(skimage.io.imread(f))
            tensor_images_list.append(img_tensor)
            tensor_labels_list.append(label_tensor)

        batch_images = torch.stack(tensor_images_list)
        batch_labels = torch.stack(tensor_labels_list)
    return batch_images, batch_labels


class ImgAug:
    def __init__(self, batch_img, value):
        """
        Augment images by batch
        :param batch_img: [batch, C, H, W]
        :param value: the degree at which the imgs are transformed 
                      (recommended 0.3 -> 0.3*32 = 9.6 pixels)

        :return: transformed batch [batch, C, H, W]
        """

        self.batch_img = batch_img
        self.value = value
        aug_method = random.randint(0, 2)
        aug_dict = {
            '0': self.__horizontal_shift(),
            '1': self.__vertical_shift(),
            '2': self.__rotate()
        }
        
        augmented_batch = aug_dict[str(aug_method)]
        self.resized_img = self.__fill(augmented_batch)
 
    def apply_trans(self):
        return self.resized_img
       
    @staticmethod
    def __fill(img):
        return T.Resize(size=(32,32), antialias=True)(img)

    def __horizontal_shift(self):
        ratio = random.uniform(-self.value, self.value)
        shift_by = int(ratio*(self.batch_img.size()[-1]))
        if shift_by < 0: # shift to the right
            shifted_batch_img = self.batch_img[:, :, :, :shift_by]
        elif shift_by > 0: # shift to the left
            shifted_batch_img = self.batch_img[:, :, :, shift_by:]
        else:
            shifted_batch_img = self.batch_img
        
        return shifted_batch_img

    def __vertical_shift(self):
        ratio = random.uniform(-self.value, self.value)
        shift_by = int(ratio*(self.batch_img.size()[-2]))
        if shift_by < 0: # shift to the upward
            shifted_batch_img = self.batch_img[:, :, :shift_by, :]
        elif shift_by > 0: # shift to the downward
            shifted_batch_img = self.batch_img[:, :, shift_by:, :]
        else:
            shifted_batch_img = self.batch_img

        return shifted_batch_img
    
    def __rotate(self):
        return T.RandomRotation(degrees=45)(self.batch_img)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
IMG_WIDTH  = 32
IMG_HEIGHT = 32

class ImageDataset(Dataset):
    def __init__(self,
                 folder_path=None, # input: image folder to be converted to dataset
                 images=None,      # input: image tensor [num_imgs, C, W, H]
                 labels=None,      # input: labels tensor [num_labels]
                 transform=None    # input: transformation method applied to images
                 ):
        self.transform = transform

        if folder_path:
            # Loading images from folder
            self.image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
            self.folder_path = folder_path
            self.images = None
            self.labels = None
        elif images is not None and labels is not None:
            # Using pre-loaded tensors
            self.images = images
            self.labels = labels
            self.image_files = None
            self.folder_path = None
        else:
            raise ValueError("Either folder_path or (images, labels) must be provided")

    def __len__(self):
        if self.image_files:
            return len(self.image_files)
        elif self.images is not None:
            return len(self.images)
        else:
            raise RuntimeError("Dataset is not properly initialized")

    def __getitem__(self, idx):
        if self.image_files:
            # Load image from disk
            image_file = self.image_files[idx]
            image_path = os.path.join(self.folder_path, image_file)
            image = Image.open(image_path).convert('RGB')
        else:
            # Get image from pre-loaded tensors
            image = self.images[idx]

        # Apply transformation to image
        if self.transform:
            image = self.transform(image)

        # Label tensor is available, add it to the dataset also
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

class PreProcessing:
    LOWER_THRESH = 50
    UPPER_THRESH = 200

    def __init__(self):
        pass

    """
        Input images are expected to be colored
    """
    def augment_data(self,
                     input_folder_path,
                     output_folder_path,
                     target_num_data):

        # Define transformations for augmentation
        augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Shift by 10% of the image size
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ])

        # Load images into a tensor
        dataset    = ImageDataset(input_folder_path, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Ensure output folder exists
        os.makedirs(output_folder_path, exist_ok=True)

        images = []

        # Collect all images in a list
        for images_batch in dataloader:
            images.append(images_batch[0])

        # images = torch.stack(images)

        # Shuffle the images
        # images = images[torch.randperm(images.size(0))]

        # Apply augmentations until target number of images is reached
        org_img_idx  = 0
        num_org_imgs = len(dataset)
        augmented_count = target_num_data - len(dataset)
        for augIdx in range(augmented_count):
            img = images[org_img_idx]
            img = transforms.ToPILImage()(img)
            augmented_image = augmentation_transforms(img)
            augmented_image_path = os.path.join(output_folder_path, f'augmented_{augIdx + 1}.png')
            augmented_image.save(augmented_image_path)

            org_img_idx = (org_img_idx + 1) % num_org_imgs
        self.save_images(self, dataloader, output_folder_path)

    def apply_preprocessing(self,
                            input_folder_path,
                            output_folder_path):
        self.create_folder(self, output_folder_path)

        for filename in os.listdir(input_folder_path):
            # Get the file extension
            _, ext = os.path.splitext(filename)

            # Check if the file is an image
            if ext.lower() in IMAGE_EXTENSIONS:
                file_path = os.path.join(input_folder_path, filename)
                image     = cv2.imread(file_path)
                processed_image = self.preprocess(self, image)

                output_filepath = os.path.join(output_folder_path, filename)
                cv2.imwrite(output_filepath, processed_image)

    def preprocess_batch(self, batch_tensor):
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

        img_res = self.preprocess(self, img_np)
        img_tensor = torch.from_numpy(img_res).unsqueeze(0)
        preprocessed_image.append(img_tensor)
        idx += 1

      preprocessed_tensor = torch.stack(preprocessed_image)
      return preprocessed_tensor

    @staticmethod
    def preprocess(self, image):
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

    @staticmethod
    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def save_images(self, dataloader: DataLoader, output_folder_path: str):
        os.makedirs(output_folder_path, exist_ok=True)

        for idx, image_batch in enumerate(dataloader):
            img = image_batch[0]
            img_pil = transforms.ToPILImage()(img)
            original_image_path = os.path.join(output_folder_path, f'original_{idx + 1}.png')
            img_pil.save(original_image_path)

class Reporter:
    def __init__(self, file_name):
        self.file_name = file_name
        self.cols = ["Epoch", "Batch ID", "Margin Loss", "Reconstruction Loss", "Total Loss", "Accuracy"]
        self.epoch, self.batch, self.marg_loss, self.recons_loss, self.total_loss, self.accuracy  = '', '', '', '', '', ''
           
    def __create_df(self):
        data = (self.epoch, self.batch, self.marg_loss, self.recons_loss, self.total_loss, self.accuracy)
        df = pd.DataFrame([data], columns=self.cols) 
        return df

    def record(self, data: dict,  sheet: str) -> None: 
        self.epoch = data['Epoch']
        self.batch = data['Batch ID']
        self.marg_loss = data['Loss']['Margin']
        self.recons_loss = data['Loss']['Recon']
        self.total_loss = data['Loss']['Total']
        self.accuracy = data["Accuracy"]
        
        if not os.path.isfile(self.file_name):
            self.__create_df().to_excel(self.file_name, sheet_name=sheet ,index=False)
        else:
            with pd.ExcelWriter(
                        self.file_name, 
                        mode="a", 
                        engine="openpyxl", 
                        if_sheet_exists="overlay") as writer:
                if sheet in pd.ExcelFile(self.file_name).sheet_names:
                    self.__create_df().to_excel(writer, sheet_name=sheet, 
                                                startrow=writer.sheets[sheet].max_row, 
                                                index=False, header=None)
                else:
                    self.__create_df().to_excel(writer, sheet_name=sheet,index=False)

"""
Extract the labeled object information from the annotated file
    input: .txt file path
    ouput: list of tuples
"""
def parse_annotation(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        
    rois = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width    = float(parts[3])
        height   = float(parts[4])
        
        # Convert to ROI (top-left x, top-left y, width, height)
        rois.append((class_id, x_center, y_center, width, height))
    
    return rois

"""
Crop the annotated object from the original image
    input: 
"""
def crop_and_save_image(
        image_file,      # image file path
        annotation_file, # annotation file path
        base_name,
        output_folder,    # destination folder
        num_classes = 51 # Vietnam Traffic Sign
        ):  
    # Read the original image
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    
    # Parse annotations
    rois = parse_annotation(annotation_file)
    
    for class_idx in range(0, num_classes):
        class_folder = os.path.join(output_folder, str(class_idx)) 
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
    
    for i, (class_id, x_center, y_center, w, h) in enumerate(rois):
        class_folder = os.path.join(output_folder, str(class_id))
        # Convert normalized coordinates to pixel values
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        w_pixel = int(w * width)
        h_pixel = int(h * height)
        
        # Compute the top-left and bottom-right points
        x1 = int(x_center_pixel - (w_pixel / 2))
        y1 = int(y_center_pixel - (h_pixel / 2))
        x2 = int(x_center_pixel + (w_pixel / 2))
        y2 = int(y_center_pixel + (h_pixel / 2))
        
        # Crop the image
        cropped_image = image[y1:y2, x1:x2]
        
        # Define the output path
        output_path = os.path.join(class_folder, f'{base_name}_{i}.jpg')
        
        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)