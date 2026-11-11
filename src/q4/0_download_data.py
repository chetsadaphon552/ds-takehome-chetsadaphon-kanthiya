"""
Problem 4 - Machine Learning: Credit Card Default Prediction
Data Download Script for UCI Default of Credit Card Clients Dataset

This script downloads the official UCI dataset (30,000 records) and extracts it
into the data/ directory, ensuring 100% reproducibility.
"""

import os
import io
import zipfile
import urllib.request


UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATA_DIR = os.path.join("src", "q4", "data")


def download_and_extract():
    """Download and extract UCI credit card dataset to src/q4/data/"""
    os.makedirs(DATA_DIR, exist_ok=True)
    target_excel = os.path.join(DATA_DIR, "default of credit card clients.xls")

    # ถ้ามีชุดข้อมูลอยู่แล้ว ให้ใช้ไฟล์เดิมและไม่ดาวน์โหลดซ้ำ
    if os.path.exists(target_excel):
        print(f"[i] Dataset already exists at: {target_excel}")
        return target_excel

    # ดาวน์โหลดและแตกไฟล์ข้อมูลจาก UCI ลงในโฟลเดอร์ของโจทย์
    print(f"Downloading UCI Credit Card dataset from:\n  {UCI_DATASET_URL} ...")
    req = urllib.request.Request(UCI_DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content = response.read()

    print(f"Downloaded {len(content):,} bytes. Extracting archive...")
    with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"[+] Dataset successfully extracted to: {target_excel}")
    return target_excel


if __name__ == "__main__":
    download_and_extract()
