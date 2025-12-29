# Traffic Sign Classifer
- This repo contains source code to train, test and validate a Deep Neural Network against [Vietnam Road Signs Dataset](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).
- To train a model, please refer to `src/TrafficSignTest.ipynb`.
- The obtained model can be coupled with a YOLO object-detection model to classify traffic signs in real-time, the detailed implementation can be found in `src/yolo_cam.py`. 

### Dataset
**1. Data filtering:** 
- The dataset contains taken pictures in the dataset contains many other objects (vehicles, trees, buildings, etc.). Therefore, to capture the traffic signs only for our classification task, some data-filtering operations are required:
  + The Vietnam Road Signs Dataset is annotated in YOLO format. Thus, the signs can be cropped from the original images using the ROI's information provided by these annotation files.
  + Function `crop_and_save_image` dedicated to signs cropping is provided in `src\utils.py`.
    
**2. Data augmentation:**
- Like any other dataset, to ensure the even distribution and to increase the amount of training data, serveral image augmentation techniques are applied. For detailed implementation, please refer to function `augment_data` in `src\utils.py`
  
**3. Image processing:**
- To improve the efficiency, images are converted into grayscale before being fed into our model.
- Other image processing modules like Gaussian Blur or Equalized Histogram to remove the noise and improve the contrast respectively.
- Thresholding can also be applied to separate the signs from background. Nonetheless, in this dataset this approach isn't feasible since images are taken at different lighting conditions and the traffic signs themselves are distinct from each other. Therefore, this module was discarded eventually. Feel free to apply this module to your dataset if possible to yield better results.

<p align="center">
    <img src="https://github.com/user-attachments/assets/542825d0-843b-4e59-8ce2-f995b106aff5" width="400">
</p>

### Architecture
- The model was observed to converge the accuray 90% benchmark faster (reduced from 70 -> less than 40 epochs) when Multiscale Convolutional Network features were integrated
  to enhance feature extraction.

  <p align="center">
    <img src="https://github.com/user-attachments/assets/7dd8c13d-7fe4-472a-9e5d-0ab1a3ef8e91" width="800">
  </p>

### Results
- The model was able to reach a peak of 98.15% accuracy at epoch 97 for the validation set, while that of the test set (with 2919 images) is 99.49%.
- As observed in the average accuracy graph, the 90% validation accuracy is reached after approximately 40 epochs. 
  
<p align="center">
    <img src="https://github.com/user-attachments/assets/704842df-d89d-4db3-8a91-b816e70263ad" width="800">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/6aaac7be-9079-494d-8e29-725545e653d9" width="800">
</p>

### References
- [vamsiramakrishnan/TrafficSignRecognition](https://github.com/vamsiramakrishnan/TrafficSignRecognition)
- [chengyangfu/pytorch-vgg-cifar10](https://github.com/chengyangfu/pytorch-vgg-cifar10).
