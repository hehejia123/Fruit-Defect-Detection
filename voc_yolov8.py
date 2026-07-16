import os
import random
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image

def convert_voc_to_yolov8(base_dir):
    """
    将VOC格式数据集转换为YOLOv8格式
    参数:
        base_dir: 数据集根目录 (包含Annotations和JPEGImages的目录)
    """
    # 设置随机种子保证可重复性
    random.seed(2022)
    
    # 定义路径
    base_dir = Path(base_dir)
    xml_dir = base_dir / "Annotations"
    img_dir = base_dir / "JPEGImages"
    
    # 检查目录是否存在
    if not xml_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"缺少Annotations或JPEGImages目录，请检查路径: {base_dir}")

    # 创建YOLOv8标准目录结构
    yolov8_dir = base_dir.parent / f"{base_dir.name}_yolov8"
    (yolov8_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolov8_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolov8_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolov8_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    img_files = [f for f in img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not img_files:
        raise ValueError(f"JPEGImages目录中没有找到图片文件: {img_dir}")

    # 打乱数据集
    random.shuffle(img_files)
    split_idx = int(len(img_files) * 0.8)  # 80%训练集

    # 类别映射字典 (根据实际类别修改)
    class_names = ["defect"]  # 替换为您的实际类别
    class_id = {name: idx for idx, name in enumerate(class_names)}

    # 处理每个图片文件
    for i, img_file in enumerate(img_files):
        # 确定训练集/验证集
        split = "train" if i < split_idx else "val"
        
        # 对应的XML文件
        xml_file = xml_dir / f"{img_file.stem}.xml"
        if not xml_file.exists():
            print(f"警告: 缺少XML标签文件 {xml_file}")
            continue

        # 获取图片尺寸
        with Image.open(img_file) as img:
            img_w, img_h = img.size

        # 解析XML并转换为YOLO格式
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            yolo_lines = []
            
            for obj in root.findall("object"):
                cls = obj.find("name").text
                if cls not in class_id:
                    print(f"警告: 发现未定义的类别 '{cls}'，已跳过")
                    continue
                    
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # 坐标归一化
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                yolo_lines.append(f"{class_id[cls]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 写入YOLO标签文件
            label_file = yolov8_dir / "labels" / split / f"{img_file.stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_lines))

            # 复制图片到对应目录
            img_dest = yolov8_dir / "images" / split / img_file.name
            shutil.copy(img_file, img_dest)

        except Exception as e:
            print(f"处理文件 {img_file} 时出错: {str(e)}")
            continue

    # 生成dataset.yaml
    yaml_content = f"""path: {yolov8_dir}
train: images/train
val: images/val
test:  # 可选测试集路径

# 类别名称
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    with open(yolov8_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"转换完成！YOLOv8格式数据集已保存到: {yolov8_dir}")
    print(f"训练集: {len(img_files[:split_idx])}张, 验证集: {len(img_files[split_idx:])}张")
    print(f"类别列表: {class_names}")

if __name__ == "__main__":
    # 使用示例 - 替换为您的实际路径
    dataset_path = r"C:\Users\17866\Desktop\ultralytics-main (1)\ultralytics-main\data\defect"
    convert_voc_to_yolov8(dataset_path)