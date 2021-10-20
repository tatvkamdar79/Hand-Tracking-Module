# Importing Necessary Libraries
import cv2
import mediapipe as mp
from time import time

class handDetector():
    def __init__(self, mode = False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initializing default parmeters and arguments
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Creating Hand Detection Object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        
        # Creating Hand Detection Drawing Object
        self.mpDraw = mp.solutions.drawing_utils

    # Function find the hands landmarks and draw if given argument to draw is true 
    def findHands(self, img, img2, draw=True, toDraw = mp.solutions.hands.HAND_CONNECTIONS):

        # Converting image to RGB since OpenCV caputers Images in BGR Format 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calling In-built MediaPipe function to process frames and detect hand and return Landmarks 
        self.results = self.hands.process(imgRGB)

        # If the argument draw is true, this loop Draws the landmarks on an alternative image img2 or give original image to draw on original image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img2, handLms, toDraw)
        return img

    # Function to return the list of positions of the landmarks in the image/frame
    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])

                # To draw circles on the landmark points on the detected hand
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lmlist

# Main function to run this module by itself or to import it in any other project
def main():
    # Initial time for FPS capture
    pTime = 0
    cTime = 0

    # Video Capturing Code
    cap = cv2.VideoCapture(0)

    # Initializing Hand Detector Object
    detector = handDetector()

    # Code to capture frames and detect hands
    while True:
        _, img = cap.read()

        # Finding Hands in the Frame
        img =  detector.findHands(img, img)

        # Getting list of positions of the landmarks the hand in the image
        lmlist = detector.findPosition(img, draw=False)

        # Printing Positions of Hand Landmarks
        if (len(lmlist)) != 0:
            print(lmlist)

        # Getting FPS
        cTime = time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        # Putting FPS on the image
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,255,0), 3)

        # Displaying the frame 
        cv2.imshow("Frame", img)
        
        # To exit the program press 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Release camera and destroy any/all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()