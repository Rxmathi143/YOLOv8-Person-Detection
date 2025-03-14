import cv2
import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(voc_dir, yolo_dir, classes_file):
    os.makedirs(yolo_dir, exist_ok=True)
    
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()
        
        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)
        
        yolo_txt_file = os.path.join(yolo_dir, xml_file.replace(".xml", ".txt"))
        
        with open(yolo_txt_file, "w") as yolo_f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    continue
                
                class_id = classes.index(class_name)
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("voc_dir", help="C:\Users\mathi\OneDrive\Desktop\project\YOLOv8-Person-Detection\datasets\labels")
    parser.add_argument("yolo_dir", help="C:\Users\mathi\OneDrive\Desktop\project\YOLOv8-Person-Detection\datasets\yolo_labels")
    parser.add_argument("classes_file", help="C:\Users\mathi\OneDrive\Desktop\project\YOLOv8-Person-Detection\datasets\classes.txt")
    args = parser.parse_args()

    convert_voc_to_yolo(args.voc_dir, args.yolo_dir, args.classes_file)