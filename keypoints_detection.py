import cv2
import mediapipe as mp


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
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
