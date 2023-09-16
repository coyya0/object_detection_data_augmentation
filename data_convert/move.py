import shutil
import os


def move_files_to_base_dir(base_dir):
    # base_dir 내의 모든 하위 디렉터리 가져오기
    subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                # 각 파일을 base_dir로 이동
                file_path = os.path.join(root, file)
                dest_path = os.path.join(base_dir, file)
                shutil.move(file_path, dest_path)

        # 이동이 완료된 후 하위 디렉터리 삭제
        shutil.rmtree(subdir_path)


# 실행 예시
base = os.getcwd()
target = os.path.join(base, "labels/train")
move_files_to_base_dir(target)  # 여기서는 실제 실행하지 않습니다.
