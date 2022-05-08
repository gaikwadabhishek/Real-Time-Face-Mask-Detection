# Face-Mask-Detection-Using-SSD-and-CNN
Detecting facemasks using SSD and CNN

## Abstract
In this project, a face mask detection model is proposed that can classify people not wearing masks in a live video feed. The system can be deployed in a network of surveillance cameras and predictions from the model can be used to notify concerned authorities in case of mask mandate violation. The model implements a deep learning architecture that has been trained on a dataset containing images of people with and without masks along with bounding box coordinates for every face present in each image. The trained model achieved 95.04% accuracy on previously unseen test data. 
Keywords: Computer vision, deep learning, face mask detection, SSD 

## Data
### Pretrained model (SSD Model for Face Detection)

[Caffe Face Detector (OpenCV Pre-trained Model)](https://www.kaggle.com/datasets/sambitmukherjee/caffe-face-detector-opencv-pretrained-model)

### Dataset

[Face Mask Detection Dataset](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset)

## Usage

### Train
Code realted to loading model, preprocessing, architecture is in 
`milestone3.ipynb`
or
`milestone3.py`

### Test
`predict.py [options] absolute/path/to/file`

Where `options` can be:
  * `photo` to predict correctly worn fask masks in a photo
  * `video` to predict correctly worn fask masks in a video 
  * `live` to predict correctly worn fask masks in a live feed through camera (here absolute/path/to/file is '0' for webcam and '1' for external camera)

For example:

    python predict.py live 0
    python predict.py video D:/video2.mp4
    python test.py image D:/test3.jpeg

## Paper

[Link](https://nodejs.org/api/vm.html)

## Results

Test Loss for Model:  0.1662766933441162
Test Accuracy for Model:  0.9408695697784424

### Model Accuracy:
![Model Accuracy](https://github.com/gaikwadabhishek/Real-Time-Face-Mask-Detection/blob/main/Results/model_accuracy.png?raw=true)

### Model Loss:
![Model Loss](https://github.com/gaikwadabhishek/Real-Time-Face-Mask-Detection/blob/main/Results/model_loss.png?raw=true)

## 
