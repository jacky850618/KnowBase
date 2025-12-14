from unstructured.partition.pdf import partition_pdf
from transformers import pipeline, DetrImageProcessor, TableTransformerForObjectDetection
import easyocr
from PIL import Image
import io
import streamlit as st
import numpy as np
import torch

@st.cache_resource
def get_image_captioner():
    """本地图片描述模型 (BLIP)"""
    try:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"加载 BLIP 失败: {e}")
        return None

@st.cache_resource
def get_table_detector():
    """本地表格检测模型 (Table Transformer)"""
    try:
        processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        return processor, model
    except Exception as e:
        st.error(f"加载 Table Transformer 失败: {e}")
        return None, None

@st.cache_resource
def get_ocr_reader():
    """本地 OCR (EasyOCR)"""
    try:
        return easyocr.Reader(['en', 'ch_sim'])  # 支持英中
    except Exception as e:
        st.error(f"加载 EasyOCR 失败: {e}")
        return None

def parse_pdf_elements(file_path: str):
    """本地解析 PDF：提取文本、表格、图片"""
    elements = partition_pdf(
        filename=file_path,
        strategy="auto",  # 自动检测元素
        infer_table_structure=True,  # 尝试表格结构
        languages=["eng", "chi_sim"]  # 支持英中
    )

    text_chunks = []
    table_chunks = []
    image_chunks = []

    for element in elements:
        if element.category == "Text":
            text_chunks.append(element.text)
        elif element.category == "Table":
            # 用本地模型增强表格解析
            table_chunks.append(parse_table(element))
        elif element.category == "Image":
            # 用本地模型生成图片描述
            image_chunks.append(parse_image(element))

    return text_chunks, table_chunks, image_chunks

def parse_table(table_element):
    """本地表格解析：Table Transformer 检测 + EasyOCR 提取"""
    processor, model = get_table_detector()
    reader = get_ocr_reader()
    if processor is None or reader is None:
        return table_element.to_dict()["text_as_html"]  # 回退 Unstructured 默认

    # 从元素提取图片
    img = Image.open(io.BytesIO(table_element.image_bytes)) if hasattr(table_element, "image_bytes") else None
    if img is None:
        return table_element.text

    # Table Transformer 检测结构
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # EasyOCR 提取文本
    img_np = np.array(img)
    cells = []
    for box in results["boxes"]:
        cell_img = img.crop(box.tolist())
        ocr_result = reader.readtext(np.array(cell_img))
        cell_text = " ".join([text for _, text, _ in ocr_result])
        cells.append(cell_text)

    # 简单重建 Markdown 表格（假设 4 列）
    md_table = "| " + " | ".join(cells[:4]) + " |\n| --- | --- | --- | --- |\n"
    md_table += "| " + " | ".join(cells[4:]) + " |"
    return md_table

def parse_image(image_element):
    """本地图片描述：BLIP 生成 caption"""
    captioner = get_image_captioner()
    if captioner is None:
        return "图片: [无法描述]"

    img = Image.open(io.BytesIO(image_element.image_bytes)) if hasattr(image_element, "image_bytes") else None
    if img is None:
        return "图片: [提取失败]"

    result = captioner(img)[0]["generated_text"]
    return f"图片描述: {result}"