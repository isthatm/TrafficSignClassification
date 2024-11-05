import cv2
import torch
import torch.utils
import vgg
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import os
import torch.nn as nn
import utils


PRINT_FREQ     = 20
HALF_PRECISION = False
TOP_k          = (1,)
ARCH           = "vgg16_bn"
RESUME         = r"C:\Users\Admin\Documents\7. Models\Vietnam\24.08.18_Take1\checkpoint_99.tar" # Path to model file
TEST_OUTPUT    = r"C:\Users\Admin\Documents\6. Data\VietnamSigns\test_pre_processing_output"
output_size = (32, 32)
# Transformation for resizing the cropped image
resize_transform = transforms.Resize(output_size)
to_pil = transforms.ToPILImage()
preprocessor = utils.PreProcessing()

def save_batch(dir, img_tensor):
    for i, tensor in enumerate(img_tensor):
        img_pil = to_pil(tensor.squeeze(0)) # remove the first dim
        file_name = f"cropped_img_{i}.png"
        img_pil.save(os.path.join(dir, file_name))

def crop_rois(roi_tensor, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    
    # Iterate through each ROI and crop the frame
    cropped_images = []
    for roi in roi_tensor:
        x_min, y_min, x_max, y_max = roi.int()  # Convert coordinates to integers
        cropped_image = frame_tensor[:, y_min:y_max, x_min:x_max]  # Crop the frame
        # Resize the cropped image to the desired size
        cropped_image_resized = resize_transform(cropped_image)

        # Add a batch dimension to get [1, c, h, w]
        # cropped_image_resized = cropped_image_resized.unsqueeze(0)
        cropped_images.append(cropped_image_resized)
    cropped_images_batch = torch.stack(cropped_images)
    
    return cropped_images_batch

def validate(input, model, criterion):
    """
    Run evaluation
    """
    # switch to evaluate mode
    # TODO: preprocess here
    model.eval()
    input = input.float()
    #print(f"input type: {input.dtype}")
    if torch.cuda.is_available():
        input  = input.cuda(non_blocking = True)
        target = target.cuda(non_blocking = True)
    if HALF_PRECISION:
        input = input.half()
    
    # compute output
    with torch.no_grad():
        output = model(input)
        #loss = criterion(output, target)
    output = output.float()

    # Determine the prediction with highest confidence
    maxk = max(TOP_k)
    _, pred = output.topk(maxk, 1, True, True)
    return pred.t()

# =============== SETUP ===============
# 1. Load the YOLOv5 model (object detection)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
weights_path = r"C:\Users\Admin\Documents\7. Models\Vietnam\24.08.24_objectdetection\best.pt"
model = YOLO(weights_path)

# 2. Load classification model
class_model_method = getattr(vgg.VGG, ARCH)
model_class = class_model_method()
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model_class.cuda() 
    checkpoint = torch.load(RESUME)
    criterion = criterion.cuda()
else:
    model_class.cpu()
    checkpoint = torch.load(RESUME, map_location=torch.device('cpu'))
    criterion = criterion.cpu()

model_class.load_state_dict(checkpoint['state_dict'])

# =============== MAIN ===============
# Input your video file (replace with your file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect objects in the frame
    results = model(frame)

    # Extract bounding boxes and class names
    for res in results:
        #xmin, ymin, xmax, ymax, confidence, class_idx = detection[:6]
        boxes = res.boxes
        probs = res.probs
        obj_rois = boxes.xyxy

        if (obj_rois.size(dim=0) > 0):
            detected_objs = crop_rois(obj_rois, frame)
            processed_batch = preprocessor.preprocess_batch(detected_objs)

            # 2. Classification
            pred = validate(processed_batch, model_class, criterion)
            # save_batch(TEST_OUTPUT, detected_objs)
        
            print(pred)
        # # Draw the bounding box
            for roi_idx, roi in enumerate(obj_rois):
                xmin, ymin, xmax, ymax = roi.int()  # Convert coordinates to integers
                xmin = xmin.item()
                ymin = ymin.item()
                xmax = xmax.item()
                ymax = ymax.item()

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = vgg.label_name_mapping[pred[0][roi_idx].item()]
                # Display the label at the top of the bounding box
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Visualize the results on the frame
        # Display the video feed with bounding boxes in a new window
        cv2.imshow('TrafficSignClassification', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()