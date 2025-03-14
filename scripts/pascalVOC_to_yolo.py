import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(voc_dir, yolo_dir, classes_file):
    # Read class names
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Ensure YOLO directory exists
    os.makedirs(yolo_dir, exist_ok=True)

    # Get all XML annotation files
    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith(".xml"):
            continue
        
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()

        # Get image dimensions
        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)
        
        # Prepare YOLO annotation
        yolo_annotations = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_names:
                continue  # Skip classes not in the list
            
            class_id = class_names.index(class_name)

            # Get bounding box coordinates
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to YOLO format (normalize values)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save YOLO annotation
        yolo_filename = os.path.join(yolo_dir, xml_file.replace(".xml", ".txt"))
        with open(yolo_filename, "w") as f:
            f.write("\n".join(yolo_annotations))

        print(f"Converted {xml_file} -> {yolo_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC XML to YOLOv8 TXT format")
    parser.add_argument("--input", required=True, help="Path to the directory containing PascalVOC XML files")
    parser.add_argument("--output", required=True, help="Path to the output directory for YOLO format annotations")
    parser.add_argument("--classes", required=True, help="Path to the classes.txt file")

    args = parser.parse_args()
    convert_voc_to_yolo(args.input, args.output, args.classes)
