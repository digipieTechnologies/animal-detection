from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
import os
import imutils


net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def detect_animal(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image at {image_path}")
        return
    image = imutils.resize(image, width=800)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
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
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if indexes is not None and len(indexes) > 0:
        if isinstance(indexes, tuple):
            indexes = indexes[0]

        indexes = indexes.flatten()
        font = cv2.FONT_HERSHEY_PLAIN
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), font, 1, color, 2)

            # Create the folder for the detected class if it doesn't exist
            class_folder = os.path.join(output_folder, label)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Save the image in the corresponding class folder
            output_path = os.path.join(class_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, image)
    else:
        print("No objects detected")


def process_images_in_folder(folder_path, output_folder_path):
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    image_files.sort()

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        detect_animal(image_path, output_folder_path)


def upload_folder(request):
    if request.method == "POST":
        if "files" not in request.FILES:
            return HttpResponse("No files uploaded", status=400)

        files = request.FILES.getlist("files")
        temp_folder = os.path.join(os.path.dirname(__file__), "temp_uploads")
        output_folder = os.path.join(os.path.dirname(__file__), "result/ani")

        # Ensure the temp folder exists
        if not os.path.exists(temp_folder):
            try:
                os.makedirs(temp_folder)
            except PermissionError as e:
                return HttpResponse(f"Permission denied: {e}", status=403)

        # Save files to the temp folder
        for file in files:
            file_path = os.path.join(temp_folder, file.name)
            with open(file_path, "wb+") as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        process_images_in_folder(temp_folder, output_folder)

        # Cleanup temp folder after processing
        for file_name in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, file_name))
        os.rmdir(temp_folder)

        return render(request, "success.html")

    return render(request, "index.html")
