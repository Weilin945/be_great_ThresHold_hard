import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from be_great.great import GReaT
from ucimlrepo import fetch_ucirepo

# 設置 Logging 以監控實驗進度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- [Block 1] Configuration: 10 Selected Datasets ---
DATASET_CONFIG = {
    # === 醫療類 (Medical) ===
    "breast_cancer": {
        "source": "uci", "id": 17, 
        "rename_map": {"Diagnosis": "diagnosis"}, 
        "target": "diagnosis"
    },
    "heart_cleveland": {
        "source": "uci", "id": 45,
        "rename_map": {"num": "target"}, 
        "target": "target"
    },
    "pima_diabetes": {
        "source": "url",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "columns": ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'],
        "target": "class"
    },
    "mammographic_mass": {
        "source": "uci", "id": 161,
        "rename_map": {"Severity": "Severity"},
        "target": "Severity"
    },
    "ilpd": {
        "source": "uci", "id": 225,
        "rename_map": {"Selector": "Selector"},
        "target": "Selector"
    },
    "heart_failure_clinical_records": {
        "source": "uci", "id": 519,
        "rename_map": {"death_event": "DEATH_EVENT"},
        "target": "DEATH_EVENT"
    },
    
    # === Parkinsons: 使用 Direct URL 並移除 ID ===
    "parkinsons": {
        "source": "url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
        "drop_columns": ["name"], 
        "target": "status"
    },

    "heart_statlog": {
        "source": "uci", "id": 145,
        "rename_map": {"class": "class"},
        "target": "class"
    },

    # === 金融類 (Finance) ===
    # [修正重點] German Credit: 改用 URL 並手動指定欄位名稱，避免變成 Attribute1~20
    "german_credit": {
        "source": "url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "columns": [
            "status", "duration", "credit_history", "purpose", "amount", 
            "savings", "employment_duration", "installment_rate", "personal_status_sex", "other_debtors", 
            "present_residence", "property", "age", "other_installment_plans", "housing", 
            "number_credits", "job", "people_liable", "telephone", "foreign_worker", 
            "class"
        ],
        "target": "class",
        # 關鍵參數: 原始檔是用空白鍵分隔的，不是逗號
        "read_csv_kwargs": {"sep": " "}
    },

    "australian_credit": {
        "source": "uci", "id": 143,
        "rename_map": {"class": "class"},
        "target": "class"
    }
}

# --- [Block 2] Data Loader (修復版: 支援 kwargs 與 自動 Header) ---
def load_dataset(name, config):
    logging.info(f"[{name}] Loading data...")
    try:
        df = None
        # 策略 A: UCI Repo
        if config["source"] == "uci":
            dataset = fetch_ucirepo(id=config["id"])
            X = dataset.data.features
            y = dataset.data.targets
            
            # 確保 y 是 DataFrame
            if isinstance(y, pd.Series):
                y = y.to_frame()

            # 避免 X 和 y 重疊
            if y is not None:
                y_cols = y.columns.tolist()
                X = X.drop(columns=[c for c in y_cols if c in X.columns], errors='ignore')
                df = pd.concat([X, y], axis=1)
            else:
                df = X
            
        # 策略 B: Direct URL
        elif config["source"] == "url":
            # [新功能] 取得額外的 read_csv 參數 (例如 German Credit 需要 sep=" ")
            kwargs = config.get("read_csv_kwargs", {})

            # 檢查是否有手動指定 columns
            if "columns" in config:
                # 像 German Credit / Pima 這種，使用 config['columns']
                df = pd.read_csv(config["url"], names=config["columns"], **kwargs)
            else:
                # 像 Parkinsons 這種有 Header 的，直接讀取
                # Pandas read_csv 會自動將重複欄位重新命名 (如 MDVP:Jitter.1)
                df = pd.read_csv(config["url"], **kwargs)

        # 通用處理 1: 移除不需要的欄位 (如 name)
        if "drop_columns" in config:
            df = df.drop(columns=config["drop_columns"], errors='ignore')

        # 通用處理 2: 重新命名
        if "rename_map" in config:
            df = df.rename(columns=config["rename_map"])
            
        # 通用處理 3: 確保 Target 欄位存在
        target_col = config["target"]
        if target_col not in df.columns:
             for col in df.columns:
                if col.lower() == target_col.lower():
                    df = df.rename(columns={col: target_col})
                    break
        
        # 通用處理 4: 強制轉數值 (Coerce)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='ignore')

        # 移除空值
        df = df.dropna()
        
        # 再次檢查並修復重複欄位 (多一層保險)
        if df.columns.duplicated().any():
            logging.warning(f"[{name}] Found duplicate columns, auto-fixing...")
            df = df.loc[:, ~df.columns.duplicated()]
        
        logging.info(f"[{name}] Loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"[{name}] Failed to load: {str(e)}")
        return None

# --- [Block 3] Experiment Runner (Strict 1:1 Generation) ---
def run_experiment(dataset_name):
    config = DATASET_CONFIG.get(dataset_name)
    if not config: return

    # 1. 建立實驗目錄
    output_dir = f"experiments/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 載入資料
    full_data = load_dataset(dataset_name, config)
    if full_data is None: return

    # 3. Train / Test Split (80% Train, 20% Test)
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
    
    # 儲存 Ground Truth 資料
    train_data.to_csv(os.path.join(output_dir, "real_train.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "real_test.csv"), index=False)
    
    logging.info(f"[{dataset_name}] Data Split: Train={len(train_data)}, Test={len(test_data)}")
    
    # 4. 初始化 GReaT 模型
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    model = GReaT(
        llm='distilgpt2',       
        batch_size=8,           
        epochs=100,             
        experiment_dir=checkpoint_dir
    )

    # 5. 模型訓練
    logging.info(f"[{dataset_name}] Start Fine-Tuning (Epochs=100)...")
    model.fit(train_data)
    logging.info(f"[{dataset_name}] Training Completed.")
    
    # 6. 生成合成資料
    n_synth = len(train_data)
    logging.info(f"[{dataset_name}] Generating {n_synth} synthetic samples...")
    
    synthetic_data = model.sample(
        n_samples=n_synth,
        temperature=0.7,    
        device="cuda",      
        max_length=512      
    )
    
    # 7. 儲存合成資料
    synth_path = os.path.join(output_dir, "synthetic.csv")
    synthetic_data.to_csv(synth_path, index=False)
    
    # model.save(checkpoint_dir) # Optional
    
    logging.info(f"[{dataset_name}] Experiment Finished. Files saved.")
    logging.info("="*60)

# --- Main Execution Loop ---
if __name__ == "__main__":
    # 10組 指定資料集
    target_datasets = [
        #"breast_cancer", 
        #"heart_cleveland", 
        #"pima_diabetes", 
        #"mammographic_mass", 
        #"ilpd", 

        #"heart_failure_clinical_records", 
        "german_credit", # 測試這個修正
        #"parkinsons",    # 以及這個
        #"heart_statlog", 
        #"australian_credit"
    ]

    print(f"Starting Benchmark on {len(target_datasets)} Datasets...")
    
    for ds_name in target_datasets:
        try:
            run_experiment(ds_name)
        except Exception as e:
            logging.error(f"Critical Failure in {ds_name}: {str(e)}")
            continue