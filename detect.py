import cv2
import csv
import numpy as np

def detect_mouth(frame, mouth_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return mouths
def detect_running(frame, prev_frame, threshold=1.5):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude of the flow vectors
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Calculate the average magnitude of the optical flow
    avg_magnitude = np.mean(flow_magnitude)

    if avg_magnitude > threshold:
        return True
    else:
        return False
def draw_rectangles(frame, rectangles):
    for (x, y, w, h) in rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def detect(video_file, max_frames=100, output_csv='animal_behavior_data.csv'):
   
    mouth_cascade = cv2.CascadeClassifier("C:/Users/91944/Downloads/haarcascade_mcs_mouth.xml")
    cap = cv2.VideoCapture(video_file)

    
    eating_frames_threshold = 15 
    eating_frames_count = 0
    frames_processed = 0
    resting_frames_count = 0
    prev_frame = None  
 
    results1 = [] 
    results2=[]
    results3=[]
    while True:
        ret, frame = cap.read()
        if not ret or frames_processed >= max_frames:
            break

        # Detect injuries based on color variations
        
        # Detect mouths in the current frame
        mouths = detect_mouth(frame, mouth_cascade)
        draw_rectangles(frame, mouths)
        # Check if mouths are detected
        if len(mouths) > 0:
            eating_frames_count += 1
        else:
            eating_frames_count = 0

        if eating_frames_count >= eating_frames_threshold:
            # Animal is eating
            result1 = "                 Animal is eating!"
        else:
            # Animal is not eating
            result1 = "                 Animal is not eating!"

        # Running detection
        if prev_frame is not None:
            is_running = detect_running(prev_frame, frame)
            if is_running:
                result2 = "Animal is running!"
            else:
                result2 = "Animal is not running!"
        else:
            result2 ="Animal is not running!"

        prev_frame = frame
        if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, frame)
                mean_diff = np.mean(frame_diff)
                if mean_diff < 45.0:  # Adjust the threshold based on your observations
                    resting_frames_count += 1
                else:
                    resting_frames_count = 0

                if resting_frames_count >=60:
                    result3 = "  Animal is resting!"
                else:
                    result3 = "  Animal is not resting!"
        
        prev_frame = frame
        
        # Show the frame with detected mouths
       
        # Show the frame with detected mouths
        cv2.imshow('Animal Eating and Running Detection', frame)

        # Store the result in the list
        results1.append(result1)
        results2.append(result2)
        results3.append(result3)
        frames_processed += 1

        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
   
    # Write the results to a CSV file
    
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame Index', '           Eating', '                Running', '                   Resting', 'Red Shade Analysis'])

        for idx, result1, result2, result3 in zip(range(frames_processed), results1, results2, results3):
            csv_writer.writerow([idx, result1, result2, result3])