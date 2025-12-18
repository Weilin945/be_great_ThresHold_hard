# filename: check_feature_importance.py
import os
import logging
import pandas as pd
import numpy as np
from custom_great import FeatureImportanceAnalyzer  # 我們只需要引用分析器

# 設置 Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. 實驗設定 (與 run_generation.py 保持一致)
# ==========================================
ROOT_DIR = "/home/panda3/Research/CT/be_great/experiments"

DATASET_META = {
    "breast_cancer": {"target": "diagnosis"},
    "heart_cleveland": {"target": "target"},
    "pima_diabetes": {"target": "class"},
    "mammographic_mass": {"target": "Severity"},
    "ilpd": {"target": "Selector"},
    "heart_failure_clinical_records": {"target": "DEATH_EVENT"},
    "parkinsons": {"target": "status"},
    "heart_statlog": {"target": "heart-disease"},
    "german_credit": {"target": "class"},
    "australian_credit": {"target": "A15"}
}

# ==========================================
# 2. 分析流程 (只算權重，不訓練)
# ==========================================
def run_analysis_only(dataset_name):
    # 1. 檢查設定
    meta = DATASET_META.get(dataset_name)
    if not meta:
        logging.warning(f"[{dataset_name}] No metadata found. Skipping.")
        return

    dataset_dir = os.path.join(ROOT_DIR, dataset_name)
    train_path = os.path.join(dataset_dir, "real_train.csv")
    
    # 2. 檢查檔案
    if not os.path.exists(train_path):
        logging.error(f"[{dataset_name}] 'real_train.csv' not found. Skipping.")
        return

    # 3. 讀取資料
    logging.info(f"[{dataset_name}] Loading real_train.csv...")
    try:
        train_data = pd.read_csv(train_path)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to read csv: {e}")
        return

    # 4. 計算特徵重要性
    target_col = meta["target"]
    logging.info(f"[{dataset_name}] Analyzing Feature Importance (Target: {target_col})...")
    
    if target_col not in train_data.columns:
        logging.error(f"[{dataset_name}] Target '{target_col}' not found.")
        return

    # 呼叫你的核心模組計算權重
    weights = FeatureImportanceAnalyzer.get_importance(train_data, target_col)
    
    # -----------------------------------------------------------
    # [核心功能] 存檔與顯示
    # -----------------------------------------------------------
    try:
        # 將字典轉為 DataFrame
        weight_df = pd.DataFrame(list(weights.items()), columns=['Feature', 'Importance'])
        # 依照重要性降序排列
        weight_df = weight_df.sort_values(by='Importance', ascending=False)
        
        # 1. 存檔
        weight_save_path = os.path.join(dataset_dir, "feature_weights.csv")
        weight_df.to_csv(weight_save_path, index=False)
        logging.info(f"[{dataset_name}] 權重表已儲存至: {weight_save_path}")
        
        # 2. 印出 Top 5
        print(f"\n--- [{dataset_name}] Top 5 Important Features ---")
        print(weight_df.head(5).to_string(index=False))
        print("-" * 40)

        # 3. 印出完整排名
        print(f"\n--- [{dataset_name}] Full Feature Importance Ranking ---")
        print(weight_df.to_string(index=False))
        print("=" * 50 + "\n")

    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to save/print weight table: {e}")

# ==========================================
# 3. 主程式執行
# ==========================================
if __name__ == "__main__":
    # 列出你想補跑權重表的資料集
    target_datasets = [
        "breast_cancer", 
        "heart_cleveland", 
        "pima_diabetes", 
        "mammographic_mass", 
        "ilpd", 

        "heart_failure_clinical_records", 
        "german_credit", # 測試這個修正
        "parkinsons",    # 以及這個
        "heart_statlog", 
        "australian_credit"
    ]

    print(f"Starting Feature Importance Analysis on {len(target_datasets)} Datasets...")
    
    for ds_name in target_datasets:
        run_analysis_only(ds_name)