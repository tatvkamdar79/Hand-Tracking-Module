# Importing Necessary Libraries
import cv2
import mediapipe as mp
import numpy as np
import time

# Initiating basic drawing styles and utils and hand detection objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class handDetector():
    def __init__(self,min_detection_confidence=0.85, min_tracking_confidence=0.65):
        # Initializing default parameters and arguments for mediapipe built in functions
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = self.min_detection_confidence, min_tracking_confidence = self.min_tracking_confidence)

    # # Function find the hands landmarks and draw using mediapipe's given drawing methods if given argument toDraw is true
    def findHands(self, img, draw = False, toDraw = mp_hands.HAND_CONNECTIONS):
        
        # Converting image to RGB since Mediapipe requires RGB images to process and OpenCV caputers Images in BGR Format 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe function to process the RGB image to detect hand and get landmarks
        self.results = self.hands.process(imgRGB)

        # If hand is detected, self.results.multi_hand_landmarks will return True
        if self.results.multi_hand_landmarks:

            # Iterating through the landmarks of each of the 20 points on the hand 
            # and using mediapipe's built in function to draw lines and points 
            # on the landmarks i.e. fingers and connections.
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
        
        return img

    # Made a function to return the list of positions of the landmarks in the image/frame and draw as per requirement
    def findPosition(self, img, handNo=0, draw=False, drawConnections=False):
        
        # Converting image to RGB since Mediapipe requires RGB images to process and OpenCV caputers Images in BGR Format.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe function to process the RGB image to detect hand and get landmarks.
        self.results = self.hands.process(imgRGB)

        # Creating an empty list (Landmark-List)
        lmlist = []
        
        # If hand is detected, self.results.multi_hand_landmarks will return True
        if self.results.multi_hand_landmarks:
            
            # If there are multiple hands, this will help indetermining which hand's landmarks to get
            # By default it is set to 0 because this module is used only for single hand
            myHand = self.results.multi_hand_landmarks[handNo]

            # Iterating through the landmarks and getting the id and positions of the hand's landmarks
            for id, lm in enumerate(myHand.landmark):
                # The hand's landmarks are normalized hence multiplying them with image dimensions to get actual co-ordinates
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
            
            # If argument to draw the connections between points is given as True
            if drawConnections:
                # Iterates through all the adjacent connections and draws a line between the points on the hand (landmarks)
                for i in mp_hands.HAND_CONNECTIONS:
                    cv2.line(img, lmlist[i[0]][1:], lmlist[i[1]][1:], (0,0,0), 3)#, cv2.LINE_AA)
            
            # If the argument to draw points (small circles) is given as True
            if draw:
                # Iterates through the points in the lmlist i.e. landmark co-ordinates and draws circles on the points
                for i in lmlist:
                    cv2.circle(img, (i[1],i[2]), 3, (206,209,0), cv2.FILLED, cv2.LINE_8)

        return np.array(lmlist), img


def main():
    # Initializing video capturing Object
    cap = cv2.VideoCapture(0)
    cap.set(3,1200)
    cap.set(4,720)

    # Initializing hand detector object
    detector = handDetector()

    # Initial time for FPS capture
    cTime, pTime = 0, 0

    while True:
        # Reading from the camera
        _, img = cap.read()

        # Getting the lmlist and drawing cinnections as well as points on the hand in the image
        lmlist, img = detector.findPosition(img, handNo=0, draw=True, drawConnections=True)

        # Code to calculate the Frames Per Second FPS
        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime

        # Flipping image about the verticcal axis
        img = cv2.flip(img,1)

        # Putting FPS display on the frame
        cv2.putText(img, str(fps), (20,70), cv2.FONT_ITALIC, 2, (0,0,0), 2, cv2.LINE_AA)
        # Displaying the Frame
        cv2.imshow('img', img)

        # If the key 'q' is pressed it will exit the while loop
        if cv2.waitKey(1)==ord('q'):
            break
    
    # Releasing the camera and destroying all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()