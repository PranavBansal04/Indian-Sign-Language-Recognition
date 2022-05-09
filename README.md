# Indian Sign Language Recognition

Youtube - https://youtu.be/MQb06t3p1TA

## Abstract

According to National Association of deaf 18 million people are deaf in India and that's a really
huge number. For most of these people sign language is the first language for communication
due to which they face a big communication barrier in communicating with other people as very
few people know the Indian Sign Language. Most of the times these people require an
interpreter who can help them communicate with other people but having an interpreter is not
always possible. This makes it important to address this problem and provide these people with
some kind of solution so that they do not feel left out and can easily mingle with the society.
Through this project I have tried to provide a solution to that problem. It includes the use of Google's Mediapipe framework which can extract useful
information from video frames and with the help of that data a machine learning model can
accurately classify them into different categories. My work started by researching about pre-existing approaches and implementing some of them and it helped me realize the limitations those solutions have and what approaches will give best results in real time. This project will
not only make communication easier for deaf and mute people, but it can also help other people
learn Sign Language faster.

## Technologies Used

<p align="left">
<a href="https://www.python.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/python-colored.svg" width="36" height="36" alt="Python" /></a>
<a href="https://code.visualstudio.com/" target="_blank" rel="noreferrer"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/768px-Visual_Studio_Code_1.35_icon.svg.png?20210804221519" width="36" height="36" alt="Visual Studio" /></a>
<a href="https://mediapipe.dev/" target="_blank" rel="noreferrer"><img src="https://mediapipe.dev/assets/img/brand.svg" width="90" height="40" alt="Mediapipe" /></a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"><img src="https://numpy.org/doc/stable/_static/numpylogo.svg" width="80" height="36" alt="Numpy" /></a>
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"><img src="https://pandas.pydata.org/static/img/pandas_white.svg" width="80" height="36" alt="Pandas" /></a>
<a href="https://matplotlib.org/" target="_blank" rel="noreferrer"><img src="https://matplotlib.org/_static/images/logo2.svg" width="90" height="36" alt="Matplotlib" /></a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"><img src="https://opencv.org/wp-content/uploads/2022/04/logo.png" width="36" height="40" alt="OpenCV" /></a>
<a href="https://jupyter.org/" target="_blank" rel="noreferrer"><img src="https://jupyter.org/assets/homepage/main-logo.svg" width="40" height="40" alt="Jupyter Notebook" /></a>

</p>


## Setup

The project does not really require any specific setup. Make sure Python 3.6 or higher is installed on your system and you have all the required packages installed such as NumPy, Pandas, Matplotlib etc. The Ipython notebook can be run on Google Colab, Azure Notebook or locally using Anaconda Navigator.
While running the final testing script, make sure that the script has access to the device's webcam for real-time video input.

<p>=> <a src="https://github.com/pranavbansal04/Indian-Sign-Language-Recognition/blob/master/signs.pdf">signs.pdf</a> file provides a guide to different signs in Indian Sign Language.</p>

## Key work done:

- Prepared an image dataset of high-resolution images consisting of several signs for different 
alphabets and number corresponding to Indian Sign Language by recording videos through mobile phone.
- Used Mediapipe framework to extract hand landmarks and other 
important features from an image or a video frame.
- Pre processed and analyzed the data to selet important features for training.
- Added more features to classify signs properly.
- Trained several classifiers with different sets of features.


## Approach

The objective is to classify different alphabets (A-Z) and digits (0-9) in Indian
Sign Language by using the Mediapipe framework. The dataset on which Machine Learning
model is trained includes the normalized coordinates of the 21 keypoints, distances between
certain sets of keypoints and the handedness (left or right). This makes a good feature space to classify all
the signs with high accuracy. Below figure depicts the 21 key points detected on a hand by the Mediapipe Framework. 

<p align="center">
    <img src="https://github.com/pranavbansal04/Indian-Sign-Language-Recognition/blob/master/images/Output-Skeleton.jpg?raw=true" width=250 height=350>
</p>

## Workflow

<p align="center">
    <img src="https://github.com/pranavbansal04/Indian-Sign-Language-Recognition/blob/master/images/workflow.PNG?raw=true">
</p>


## Results

I was successfully able to train a Random Forest classifier with 98 % accuracy and it can 
classify 12 signs with 100% accuracy. Moreover, the model works really smoothly 
without requiring much resources in real-time and without any lag which makes it more usable.


The plot below depicts the accuracy achieved throguh different models:


<p align="center">
    <img src="https://github.com/pranavbansal04/Indian-Sign-Language-Recognition/blob/master/images/compare.png?raw=true" width=400 height=300>
</p>

