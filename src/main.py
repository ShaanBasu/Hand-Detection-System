"""
Main Application: Hand Detection
This script captures video from webcam and displays hand landmarks in real time
"""

import cv2
import sys
from hand_tracker import HandTracker 

"""
This function runs the real time hand detection
"""
def main():
    #Initialise webcam
    cap = cv2.VideoCapture(0) #Here the 0 means for the default camera if you have more cameras then we use 1,2 depending on the no. of camras
    
    #We then check if the camera was opened properly
    if not cap.isOpened():
        print("Error: Camera Could Not be Open.")
        sys.exit(1)
    
    #We then create an instance of our handtracker class
    detector = HandTracker(
        static_image_mode= False,
        max_num_hands= 2,
        min_detection_confidence= 0.7,
        min_tracking_confidence= 0.5
    )
    
    #Then we get framerate for the display
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("Starting hand detection...")
    print("Esc to exit")
    print("Press s to save a frame")
    
    frame_count=0;
    
    #We then start the main loop for the camera
    while True:
        ret, frame = cap.read() #Here ret is a boolean which indicates if the frame was successfully captured
        
        if not ret:
            print("Error: failed to read from webcam")
            break
        
        frame_count += 1 
        
        # Then we detect hand landmarks in the current frame
        results = detector.process(frame)
        
        #Then we draw the landmarks on the frame/hand 
        frame = detector.draw(frame, results)
        
        #Finally we display the frame information on the video
        
        cv2.putText(
            frame,                            #image to draw on
            f"Frame: {frame_count}",          #text to display
            (10,30),                          #position of the text
            cv2.FONT_HERSHEY_SIMPLEX,         #font type
            1,                                #font scale
            (0,255,0),                          #Color:green
            2                                  #line thickness
        )  
        
        # We also display the hand count if hands are detected
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            handedness = detector.get_handedness(results)
            
            hand_text = f"Hands Detected: {num_hands}"
            
            if handedness: 
                hand_text += ", " .join(handedness)
                
            cv2.putText(
                frame,
                hand_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # we display the annotated frame in a window    
        cv2.imshow("Hand Detection- Press Esc to exit" , frame)
        
        #Then we wait for keyboard input 
        key = cv2.waitKey(1) & 0xFF
        
        #We handle the keyboard inputs
        
        if key == 27: #The esc key
            print("Exiting...")
            break
        elif key == ord('s'):
            filename = f"hand_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"frame saved as {filename}")
            
    # Then we close the webcame and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    print("Hand detection stopped")

if __name__ == "__main__":
    """
    This line ensures main() only runs when you execute this file directly.
    It won't run if this file is imported as a module in another script.
    """
    main()
            