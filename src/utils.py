import pickle
import random
import os
import torchvision.transforms as T
import pandas as pd
import shutil
import skimage.io

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
    
    ---Params---

    file: path to pickle file

    writeColumns (array): variables to be saved to pickle file

    """
    if writeColumns is None:
        with open(file, mode="rb") as f:
            dataset = pickle.load(f)
            return tuple(map(lambda col: dataset[col], ['images', 'labels'])) # lambda(col) where columns are the inputs
    else:
        with open(file, mode="wb") as f:
            dataset = pickle.dump({"images": writeColumns[0], "labels": writeColumns[1]}, f)
            print("Data is saved in", file)

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        # label_dir contains 61 catefories paths
        label_dir = os.path.join(data_dir, d)

        # list subdirectories within each of the 61 categories
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".png")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
    return images, labels


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
