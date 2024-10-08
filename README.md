# Traffic Sign Classifer
- This repo contains source code to train, test and validate a Deep Neural Network against [Vietnam Road Signs Dataset](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).
- To train a model, please refer to `src/TrafficSignTest.ipynb`.
- The obtained model can be coupled with a YOLO object-detection model to classify traffic signs in real-time, the detailed implementation can be found in `src/yolo_cam.py`. 

# Architecture
- Before feeding images into the model, serveral image processing technqiues are implemented (gray scale conversion, thresholding, Gaussian blur, etc.) to separate traffic signs from background.
- The model was observed to converge the accuray 90% benchmark faster (reduced from 70 -> less than 40 epochs) when Multiscale Convolutional Network features were integrated
  to enhance feature extraction.

  <p align="center">
    <img src="https://github.com/user-attachments/assets/7dd8c13d-7fe4-472a-9e5d-0ab1a3ef8e91" width="800">
  </p>

# Results
- The model was able to reach a peak of 98.15% accuracy at epoch 97 for the validation set, while that of the test set (with 2919 images) is 99.49%.
- As observed in the average accuracy graph, the 90% validation accuracy is reached after approximately 40 epochs. 
  
<p align="center">
    <img src="https://github.com/user-attachments/assets/704842df-d89d-4db3-8a91-b816e70263ad" width="800">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/6aaac7be-9079-494d-8e29-725545e653d9" width="800">
</p>

# References
- This work is inspired by [vamsiramakrishnan/TrafficSignRecognition](https://github.com/vamsiramakrishnan/TrafficSignRecognition) and [chengyangfu/pytorch-vgg-cifar10](https://github.com/chengyangfu/pytorch-vgg-cifar10).
