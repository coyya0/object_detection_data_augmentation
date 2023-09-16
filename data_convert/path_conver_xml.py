import os
import xml.etree.ElementTree as ET

def update_xml_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 폴더명 ,경로명 수정
    folder_tag = root.find(".//folder")
    path_tag = root.find(".//path")

    if folder_tag is not None and path_tag is not None:
        # 현재 XML 파일의 폴더 경로
        current_folder = os.path.dirname(xml_path)

        # folder 태그 업데이트
        folder_tag.text = current_folder

        # path 태그 업데이트(파일명 유지, 경로만 변경)
        filename = os.path.basename(path_tag.text)
        path_tag.text = os.path.join(current_folder, filename)

        # 변경된 내욜을 XML 파일에 저장
        tree.write(xml_path)

def main(directroy):
    for foldername, subfolders, filenames in os.walk(directroy):
        for filename in filenames:
            if filename.endswith('.xml'):
                update_xml_file(os.paht.join(foldername, filename))

if __name__ == "__main__":
    main("D:/bk/dataset_Aihub")
    # main(os.getcwd())