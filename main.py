import cv2
import tkinter as tk
from tkinter import filedialog
import os
import csv
import numpy as np
from PIL import Image, ImageTk
from detect import detect
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def display_csv_in_html(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        next(csv_reader)
        next(csv_reader)
        next(csv_reader)
        for row in csv_reader:
            data.append(row)

    html_content = """
    <html>
    <head>
        <title>Detected Behavior</title>
        <style>
            body {
                background-color:#C7FFED;
                font-family: Arial, sans-serif;
                padding: 20px;
            }
            
            h1 {
                color:#007566;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #007566;
            }
            tr:nth-child(even) {
                background-color: #589A8D; 
            }
            tr:nth-child(odd) {
                background-color: #8FC1B5; 
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Detected Behavior</h1>
        <table>
            <thead>
                <tr>
                    <th>Frame Index</th>
                    <th>Eating</th>
                    <th>Running</th>
                    <th>Resting</th>
                    
                </tr>
            </thead>
            <tbody>
    """
    for row in data:
        html_content += "<tr>"
        for item in row:
            html_content += f"<td>{item}</td>"
        html_content += "</tr>"

    html_content += """
            </tbody>
        </table>
        <div style="width: 300px; height: 300px;">
    <canvas id="pieChart" width="150" height="150"></canvas>
</div>
        <button onclick="displayPieChart()" style="font-size: 16px; color:#007566; padding: 10px 20px; margin-top: 10px;">View Pie Chart Analysis</button>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            function displayPieChart() {
    // Parse the CSV data to count occurrences of each behavior
    var eatingCount = 0;
    var runningCount = 0;
    var restingCount = 0;

    var table = document.querySelector('table tbody');
    var rows = table.querySelectorAll('tr');

    rows.forEach(function(row) {
        var columns = row.querySelectorAll('td');
        if (columns.length === 4) {
            if (columns[1].textContent.includes('Animal is eating')) {
                eatingCount++;
            } if (columns[2].textContent.includes('Animal is running')) {
                runningCount++;
            } if (columns[3].textContent.includes('Animal is resting')) {
                restingCount++;
            }
        }
    });

    var data = {
        labels: ['Eating', 'Running', 'Resting'],
        datasets: [{
            data: [eatingCount, runningCount, restingCount],
            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
        }]
    };

    var ctx = document.getElementById('pieChart').getContext('2d');
    var pieChart = new Chart(ctx, {
        type: 'pie',
        data: data,
        options: {
            title: {
                display: true,
                text: 'Behavior Analysis Summary'
            }
           
        }
    });
}
        </script>
    </body>
    </html>
    """

    with open("csv_contents.html", "w") as html_file:
        html_file.write(html_content)

    webbrowser.open("csv_contents.html")
def load_yolo_model():
    cfg_path = os.path.abspath("C:/Users/91944/Downloads/detection/detection/yolov3.cfg")
    weights_path = os.path.abspath("C:/Users/91944/Downloads/detection/detection/yolov3.weights")
    net = cv2.dnn.readNet(weights_path, cfg_path)
    classes = []
    with open("C:/Users/91944/Downloads/detection/detection/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers
def detect_color_changes(video_path, min_contour_area=1000):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to HSV (Hue, Saturation, Value) color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the color range you want to detect
        lower_color_bound = np.array([0, 50, 50])  # Adjust these values based on the color you want to detect
        upper_color_bound = np.array([10, 255, 255])  # Hue values for red color

        # Create a mask using the color range
        mask = cv2.inRange(hsv_frame, lower_color_bound, upper_color_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around detected areas with unusual color
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display the frame with color patches highlighted
        cv2.imshow('Color Analysis(Possibilities of injury)', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def display_csv_contents(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)

    display_window = tk.Toplevel()
    display_window.title("CSV Contents")
    text_widget = tk.Text(display_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.pack(fill=tk.BOTH, expand=True)

    for row in data:
        text_widget.insert(tk.END, ", ".join(row) + "\n")

def detect_animal_behavior(video_file):
    net, classes, output_layers = load_yolo_model()
    cap = cv2.VideoCapture(video_file)
    
    behavior_labels = {
        0: 'Playing',
        1: 'Eating',
        # Add more behavior labels here...
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Non-maximum suppression to eliminate overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                behavior_label = behavior_labels[class_ids[i]] if class_ids[i] in behavior_labels else 'Unknown'
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f} ({behavior_label})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Animal Behavior Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detect_color_changes(video_file)

def start_detection():
    video_file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if video_file:
        detect_animal_behavior(video_file)
        # Open a new thread for detecting animal behavior and tracking
        detect(video_file)
        #display_csv_contents('animal_behavior_data.csv')
        display_csv_in_html('animal_behavior_data.csv')
        

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Animal Behavior Tracking System")
    root.geometry("400x400")
    root.config(bg="black")
    img_label = tk.Label(root)
    img_label.pack(pady=10)
    
    # Load the image/photo and display it on the label
    image_path = "C:/Users/91944/Downloads/images.jpg"  # Replace with the path to your image file
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Resize the image to fit the label
    photo = ImageTk.PhotoImage(img)
    img_label.config(image=photo)
    img_label.image = photo

    detect_btn = tk.Button(root, text="Upload Video",  font=("Arial", 18),command=start_detection, bg="gray", fg="white")
    detect_btn.pack()

    root.mainloop()