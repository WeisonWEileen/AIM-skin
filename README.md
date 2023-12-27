# FC-AIMS-tactile-dataset
Dynamic tactile perception based on FC-AIMS platform

Dataset download link: https://drive.google.com/drive/folders/1Ehpyk6XMn0UgFS-sOneyP3ZCYR78MxMv?usp=sharing.


# Explanation of hardware circuit
A parallel digital orthogonal demodulator based on FPGA was designed to calculate the capacitance of array nodes under frequency encoding. Multi channel parallel demodulation can meet the high-speed response of sensor arrays. 
The hardware and digital circuits for frequency encoding and decoding will be organized and open-source in the later stage.

# Spatiotemporal Touch Gestures Learning
This project reports a spatiotemporal learning architecture that learns 10 gestures' features using data from our artificial ion mechanoreceptor skin (AIM-skin).
We designed 2D convolutional (C2D), 3D convolutional (C3D), and 3D residual convolutional networks (Res3D) respectively to classify and perceive the dynamic tactile sensation of sensors.
***
## System requirements
### OS Requirements: 
This project is supported for Windows, macOS and Linux and was run on Linux with GPU.
### Python Requirements: 
The project was compiled using Python (Python 3.9.0) and four NVIDIA T4 tensor core GPUs for model inference.
## Installation guide
Some Python packages are required for the compilation.
1. numpy (version 1.24.3)
2. torch (version 2.0.0). 
Torch package provides main APIs to build our dataset, model and training progress.
***
## Instructions to run the project
1. Please download all data & codes and then unzip all the folders and place them in the same folder. Dataset download link: https://drive.google.com/drive/folders/1Ehpyk6XMn0UgFS-sOneyP3ZCYR78MxMv?usp=sharing.
2. Run the train.py using Pycharm or other supported softwares or sh do_train.sh.
Make sure that you have downloaded all the required packages and used Python with version >=3.8.  
To train the Res3D model, we uses Adam on 4 GPUs with a mini-batch size of 60 examples. The initial learning rate is 0.001 and is divided by 10 every 20 iterations, finishing at 50 iterations (You can modify the parameters).
3. See the training results on the Validation set. (You may see the results on the test set by running the test.py.)  
Running time for each epoch is close to 10s on our GPU. The whole training progress might take 8 minutes depending on your hardware.
***
## Dataset Description
The structure of our dataset is shown as below.  
0Alldata  
&emsp;1Traindata  
&emsp;&emsp;Including 15 participants  
&emsp;2Validationdata  
&emsp;&emsp;Including 2 participants  
&emsp;3Testdata  
&emsp;&emsp;Including 3 participants  
  
Each participant's data structure:  
label+participant's name  
&emsp;1touch  
&emsp;&emsp;touch classes  
&emsp;&emsp;&emsp;four tries' packages (Each includes 2 videos and 1 .npz touch data)  

&emsp;We recruited 20 participants (14 males and 6 females) to take part in our touch experiments. As an artificial skin, our sensor can capture
dynamic tactile information in the experiment. Each subject performed 10 different dynamic touch gestures, with each gesture being recorded four times for data acquisition, and the duration of each touch was 10 seconds. The gestures of *hit, stroke, rub, tap, poke, press, scratch, pat, circle, and put* were performed by the 20 subjects.  
&emsp;So, a total of 800 touch gesture samples were collected in the dataset. And each .npz touch data collected from our I-skin has the size of 5000×8×8 because the sensor array is 8×8 and the collecting frequency is 500 frames per second.

***
## Model Description
&emsp;An efficient deep 3D Residual ConvNet architecture (shown below) based on the Pytorch framework was constructed for spatiotemporal feature learning, which adopts ResNet as the backbone network by expanding 2D convolutions into 3D convolutions.  
(picture of resnet)  
&emsp;Res3D is composed of multiple basic residual blocks, which include shortcut connections that bypass signals between layers. Inside the residual block are two convolution layers with the size of 3 × 3 × 3, after which batch normalization and ReLU layers are added. This learned tactile output of ResNet can be reviewed as transferable semantic features, forming a new 512D vector, which is used as the input of the fully connected neural network for final decision learning.
***
Finally, please let us know if you have any comments or suggestions by using our codes & dataset and we would love to hear from you. You can contact us through: 12231066@mail.sustech.edu.cn or 12012508@mail.sustech.edu.cn.
