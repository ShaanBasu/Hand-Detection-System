"""
Hand Tracker Module
"""

import cv2
import mediapipe as mp
from typing import Tuple, Optional


""" Creating a wrapper class for mediapipie hands detection
    This class will initialise the media pipe hand detector
    process video frames to detect hand landmarks
    and draw the detected landmarks on frames
"""
class HandTracker:
    
    def __init__(self, static_image_mode = False, max_num_hands: int=2,min_detection_confidence: float=0.7, min_tracking_confidence: float=0.5):
        
        #Initialise MediaPipe Hands Solution(we load the pre trained machine-learning model as an object)
        self.mp_hands= mp.solutions.hands;
        
        #Then we create the tune the model with specified parameters for our project
        self.hands= self.mp_hands.Hands(
            static_image_mode = static_image_mode,
            max_num_hands = max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence= min_tracking_confidence
        )
        
        #We initilise drawing utlities on our hands for visualising the landkmarks
        self.mp_draw= mp.solutions.drawing_utils
        
        #Then we store the configuration in instance variables for reference and usage later on
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        
    
    
        """
        The following function processes a single video frame to detect hand landmarks
        and returns an object containing all the analysis output of the video frame
        """
    def process(self, frame: cv2.Mat) -> object:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Open cv is default BGR format so we have to convert the fram into rgb values
        
        results = self.hands.process(img_rgb)
        
        return results
    
    
    """
    The following function draws detected hand landmarks and connections on the frame
    Landmarks are labeled 0-20 (see hand_landmarks.proto in MediaPipe docs):
        - 0: Wrist
        - 1-4: Thumb (CMC, MCP, IP, TIP)
        - 5-8: Index finger (MCP, PIP, DIP, TIP)
        - 9-12: Middle finger (MCP, PIP, DIP, TIP)
        - 13-16: Ring finger (MCP, PIP, DIP, TIP)
        - 17-20: Pinky finger (MCP, PIP, DIP, TIP)
        
    It then returns the image in cv2.Mat format
    """
    def draw(self, frame: cv2.Mat, results: object) ->cv2.Mat:
        
        if results.multi_hand_landmarks: #Checks if there are any hands detected
            #Iterates through each hand detected
            for hand_landmarks in results.multi_hand_landmarks:
                #Draws landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame
            
    
    
    """
    The following function extracts raw hand landmarks data from the results object
    This method will be useful when we nned to extracte coordinates for gesture classification
    and calculate distance between points/fingers and analyse hand movements in 3D
    """
    def get_hand_landmarks(self, results: object) -> Optional[list]:
    
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        
        return None
    
    
    """
    This function obtains the information which hand is left/right and returns the result in a list 
    """
    def get_handedness(self, results: object) -> Optional[list]:
        
        if results.multi_handedness:
            return [hand_info.classification[0].label
                    for hand_info in results.multi_handedness]
        
        return None