import os
import xml.etree.ElementTree as ET
import yaml


def convert_to_yolo_format(bbox, img_width, img_height):
    x_center = (bbox["xmin"] + bbox["xmax"]) / 2.0
    y_center = (bbox["ymin"] + bbox["ymax"]) / 2.0
    width = bbox["xmax"] - bbox["xmin"]
    height = bbox["ymax"] - bbox["ymin"]

    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return x_center, y_center, width, height


def parse_custom_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def convert_and_save_xml_to_yolo(xml_path, base_dir, class_names):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 이미지의 너비와 높이 가져오기
        img_width = int(root.find("size").find("width").text)
        img_height = int(root.find("size").find("height").text)

        # YOLO 형식의 레이블 정보를 저장할 리스트
        yolo_labels = []

        # XML 정보를 파싱하여 객체 정보 추출
        for member in root.findall("object"):
            class_name = member.find("name").text
            if class_name not in class_names:
                print(
                    f"Warning: Class '{class_name}' in {xml_path} not found in custom.yaml. Skipping..."
                )
                continue
            class_id = class_names.index(class_name)

            # ... [나머지 코드는 그대로 유지]

        # 변경된 부분: yolo_labels에 저장될 경로를 설정합니다.
        relative_path = os.path.relpath(xml_path, base_dir)
        output_path = os.path.join(
            base_dir,
            relative_path.split(os.sep)[0],  # 'train' or 'val'
            "yolo_labels",
            os.sep.join(relative_path.split(os.sep)[2:]).replace(".xml", ".txt"),
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(yolo_labels))

        return output_path

    except Exception as e:
        print(f"Error processing {xml_path}. Error: {e}")
        return None


def process_and_save(base_dir, class_names):
    all_txt_paths = []
    for phase in ["train", "val"]:
        labels_dir = os.path.join(base_dir, phase, "labels")
        images_dir = os.path.join(base_dir, phase, "images")
        yolo_labels_dir = os.path.join(base_dir, phase, "yolo_labels")
        for class_folder in os.listdir(labels_dir):
            class_labels_dir = os.path.join(labels_dir, class_folder)
            class_images_dir = os.path.join(images_dir, class_folder)
            class_yolo_labels_dir = os.path.join(yolo_labels_dir, class_folder)
            os.makedirs(class_yolo_labels_dir, exist_ok=True)
            for xml_file in os.listdir(class_labels_dir):
                if xml_file.endswith(".xml"):
                    jpg_file = xml_file.replace(".xml", ".jpg")
                    if os.path.exists(os.path.join(class_images_dir, jpg_file)):
                        xml_path = os.path.join(class_labels_dir, xml_file)
                        output_path = convert_and_save_xml_to_yolo(
                            xml_path, class_yolo_labels_dir, class_names
                        )
                        all_txt_paths.append(output_path)
    return all_txt_paths


base_dir = os.getcwd()
if __name__ == "__main__":
    class_names = parse_custom_yaml("../yolov5/data/custom.yaml")
    process_and_save(base_dir, class_names)
