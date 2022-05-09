# Indian Sign Language Recognition

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


<img src="https://github.com/pranavbansal04/Indian-Sign-Language-Recognition/blob/master/images/Output-Skeleton.jpg?raw=true" width=150 height=280>

## Workflow


![image](https://github.com/pranavbansal04/Indian-Sign-Language-Recognition/blob/master/images/workflow.PNG?raw=true)
