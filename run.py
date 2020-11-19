import cv2
import mediapipe as mp

import joblib 

import time
import wordninja
from spellchecker import SpellChecker

def new_cord(old_cord,new_origin):
    orgx,orgy = new_origin
    x,y = old_cord
    return [x-orgx,y-orgy]

def distance(x1,y1,x2,y2):
    distance = ((x1-x2)**2+(y1-y2)**2)**0.5
    return round(distance,3)


model = joblib.load("random_forest.joblib")


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

### For static images:
##hands = mp_hands.Hands(
##    static_image_mode=True,
##    max_num_hands=2,
##    min_detection_confidence=0.7)
##for idx, file in enumerate(file_list):
##  # Read an image, flip it around y-axis for correct handedness output (see
##  # above).
##  image = cv2.flip(cv2.imread(file), 1)
##  # Convert the BGR image to RGB before processing.
##  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
##
##  # Print handedness and draw hand landmarks on the image.
##  print('handedness:', results.multi_handedness)
##  if not results.multi_hand_landmarks:
##    continue
##  annotated_image = image.copy()
##  for hand_landmarks in results.multi_hand_landmarks:
##    print('hand_landmarks:', hand_landmarks)
##    mp_drawing.draw_landmarks(
##        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
##  cv2.imwrite(
##      '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(image, 1))
##hands.close()

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

prev = None

while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

  # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

  # image = cv2.resize(image,(540,960))
  
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:

    handedness = results.multi_handedness[0].classification[0].label

    points = []

    for hand_landmarks in results.multi_hand_landmarks:
      for i in range(21):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        points.append([x,y])
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    org_transform = []
    new_org = points[0]
    
    for i in range(1,len(points)):
        point = points[i]
        temp = new_cord(point,new_org)
        org_transform.append(temp[0])
        org_transform.append(temp[1])
    

    distances = []
    
    p1=points[0]
    for j in range(1,len(points)):
        p2=points[j]
        distances.append(distance(p1[0],p1[1],p2[0],p2[1]))

    flag = 0

    if("right" in handedness):
      flag=1

    # print(org_transform)

    cur = model.predict([org_transform+distances+[flag]])

    
    if(prev is None):
      prev = cur
      print(cur[0],end="")
    
    elif(prev!=cur):
      print(cur[0],end="")
      prev = cur


  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
