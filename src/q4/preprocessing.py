"""
Q4 preprocessing shared by the EDA, modelling, and evaluation notebooks.

สรุป: อ่านและทำความสะอาดข้อมูล UCI, สร้างตัวแปรธุรกิจ 6 ตัว และเลือก
feature/target ให้ทุก notebook ใช้กติกาเดียวกัน
"""

import numpy as np
import pandas as pd


def load_and_clean_data(filepath='data/default of credit card clients.xls'):
    """
    โหลดข้อมูลจาก UCI และทำความสะอาด
    
    Returns:
        pd.DataFrame: ข้อมูลที่ทำความสะอาดแล้ว
    """
    # โหลดไฟล์ UCI โดยใช้แถวที่สองเป็นหัวตาราง
    df = pd.read_excel(filepath, header=1)
    
    # เปลี่ยนชื่อ target column
    df = df.rename(columns={'default payment next month': 'default'})
    
    # รวมรหัสการศึกษาที่ไม่อยู่ในคำอธิบายให้เป็นกลุ่ม Other
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    
    # รวมรหัสสถานภาพสมรสที่ไม่อยู่ในคำอธิบายให้เป็นกลุ่ม Other
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    
    # ตัด ID ที่ไม่ควรใช้ทำนายออก
    df = df.drop(columns=['ID'])
    
    return df


def engineer_features(df):
    """
    สร้าง 6 features ใหม่ตาม business logic
    
    Features:
    1. UTIL_RATE: อัตราการใช้วงเงิน (ยอดค้าง/วงเงิน)
    2. PAY_RATIO: อัตราการจ่าย (จ่าย/ค้าง)
    3. AVG_DELAY: ความล่าช้าเฉลี่ย 6 เดือน
    4. NUM_DELAYS: จำนวนเดือนที่ล่าช้า
    5. BILL_TREND: แนวโน้มหนี้ (เพิ่ม/ลด)
    6. PAY_VOLATILITY: ความผันผวนการจ่าย
    
    Returns:
        pd.DataFrame: ข้อมูลเดิม + 6 features ใหม่
    """
    df = df.copy()
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    
    # 1. อัตราการใช้วงเงิน (จำกัด 0-2)
    df['UTIL_RATE'] = np.clip(
        df['BILL_AMT1'] / np.maximum(df['LIMIT_BAL'], 1), 0, 2
    )
    
    # 2. อัตราการจ่าย (จำกัด 0-5)
    df['PAY_RATIO'] = np.clip(
        df['PAY_AMT1'] / np.maximum(df['BILL_AMT1'], 1), 0, 5
    )
    
    # 3. ความล่าช้าเฉลี่ย
    df['AVG_DELAY'] = df[pay_cols].mean(axis=1)
    
    # 4. นับจำนวนเดือนที่ล่าช้า (PAY > 0)
    df['NUM_DELAYS'] = (df[pay_cols] > 0).sum(axis=1)
    
    # 5. แนวโน้มหนี้ (บวก = หนี้เพิ่ม)
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    
    # 6. ความผันผวนการจ่าย (std)
    df['PAY_VOLATILITY'] = df[pay_amt_cols].std(axis=1)
    
    # แทนที่ NaN/Inf (เฉพาะ numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def select_features(df, include_sex=True, include_education=True, include_marriage=False):
    """
    สร้าง candidate feature pool สำหรับการเปรียบเทียบและคัดเลือกใน notebook 2.
    
    Args:
        df: DataFrame ที่มี features ทั้งหมด
        include_sex: รวม SEX (สำหรับ fairness audit)
        include_education: รวม EDUCATION  
        include_marriage: รวม MARRIAGE (optional)
    
    Returns:
        X: Candidate features (19 columns by default; notebook 2 selects the final subset)
        y: Target (default)
    """
    # Tier 1: payment history, credit limit, bill/payment amounts, and age
    tier1_features = [
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',  # ประวัติชำระ (6)
        'LIMIT_BAL',                                             # วงเงิน (1)
        'BILL_AMT1', 'BILL_AMT6',                               # ยอดค้าง (2)
        'PAY_AMT1',                                             # ยอดจ่าย (1)
        'AGE'                                                    # อายุ
    ]
    if include_education:
        tier1_features.append('EDUCATION')
    
    # Tier 2: Engineered Features (6)
    tier2_features = [
        'UTIL_RATE',       # อัตราใช้วงเงิน
        'PAY_RATIO',       # อัตราการจ่าย
        'AVG_DELAY',       # ความล่าช้าเฉลี่ย
        'NUM_DELAYS',      # จำนวนเดือนที่ล่าช้า
        'BILL_TREND',      # แนวโน้มหนี้
        'PAY_VOLATILITY'   # ความผันผวนการจ่าย
    ]
    
    # Tier 3: Fairness Features (1)
    tier3_features = []
    if include_sex:
        tier3_features.append('SEX')
    if include_marriage:
        tier3_features.append('MARRIAGE')
    
    # Candidate pool defaults to 12 core + 6 engineered + SEX for subgroup audit.
    selected_features = tier1_features + tier2_features + tier3_features
    
    X = df[selected_features].copy()
    y = df['default'].copy()
    
    return X, y


def get_feature_names(include_sex=True, include_education=True, include_marriage=False):
    """Return the candidate feature pool; notebook 2 selects the final model subset."""
    # Tier 1: core payment, credit, bill/payment, and age features
    tier1 = [
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',  # Payment history (6)
        'LIMIT_BAL',                                             # Credit limit (1)
        'BILL_AMT1', 'BILL_AMT6',                               # Bill amounts (2)
        'PAY_AMT1',                                             # Payment amount (1)
        'AGE'                                                    # Age
    ]
    if include_education:
        tier1.append('EDUCATION')
    
    # Tier 2: Engineered features (6)
    tier2 = [
        'UTIL_RATE', 'PAY_RATIO', 'AVG_DELAY',
        'NUM_DELAYS', 'BILL_TREND', 'PAY_VOLATILITY'
    ]
    
    # Tier 3: Fairness features (1)
    tier3 = []
    if include_sex:
        tier3.append('SEX')
    if include_marriage:
        tier3.append('MARRIAGE')
    
    return tier1 + tier2 + tier3


if __name__ == "__main__":
    # Test the pipeline
    print("Testing preprocessing pipeline...")
    
    # Load and clean
    df = load_and_clean_data()
    print(f"[OK] Loaded: {df.shape}")
    
    # Engineer features
    df = engineer_features(df)
    print(f"[OK] Engineered: {df.shape}")
    
    # Select features
    X, y = select_features(df, include_sex=True, include_education=True, include_marriage=False)
    print(f"[OK] Selected: X={X.shape}, y={y.shape}")
    
    # Check for NaN/Inf
    n_nan = X.isnull().sum().sum()
    n_inf = np.isinf(X).sum().sum()
    print(f"[OK] Quality: {n_nan} NaN, {n_inf} Inf")
    
    print(f"\n[OK] Feature list:\n{X.columns.tolist()}")
