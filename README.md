# Live-face-recognition-for-security-purpose

  ## Brief Description
  This project uses [facenet](https://arxiv.org/pdf/1503.03832.pdf) to recognise the person's face. If the person is not         identified as an authorized person it triggers an alarm to start.Inception network as described in the paper is used as
  128-d feature extractor and we used [Harr Cascade Classifier](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) for live face detection and passed the roi extracted (after resizing it to 96x96, since the model was trained on the input of size 96x96)  by the face detector to the inception network to extract the 128-d feature(embedding) and then we set the threshold for after calculating the L2 distance.
  
 ## Team Members:
   - [Mukul Ranjan](https://github.com/mukul54/)
   - [Jayant Prakash Singh](https://github.com/jayantp07)
   - [Chandan Govind Agrawal](https://github.com/chan4899)
       
