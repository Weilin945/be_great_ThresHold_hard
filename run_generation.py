# filename: run_generation.py
import os
import logging
import time
import pandas as pd
import numpy as np
import torch
from custom_great import WeightedGReaT, FeatureImportanceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. 路徑與參數設定
# ==========================================
# 原始資料路徑 (Input)
ROOT_DIR = "/home/panda3/Research/CT/be_great_Threshold_hard/experiments"

# [修改重點] 設定輸出路徑 (Output) - 與 experiments 同層級
# os.path.dirname(ROOT_DIR) 會回到上一層 (.../be_great_Threshold)
# 然後再建立 filtered_experiments 資料夾
OUTPUT_ROOT = os.path.join(os.path.dirname(ROOT_DIR), "filtered_experiments")

# 特徵重要性門檻
IMPORTANCE_THRESHOLD = 0.01 

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

def save_experiment_report(output_dir, dataset_name, stats):
    """將實驗數據寫入人類可讀的 TXT 報告"""
    report_path = os.path.join(output_dir, "experiment_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"==================================================\n")
        f.write(f"   Experiment Report: {dataset_name}\n")
        f.write(f"==================================================\n\n")
        f.write(f"[1] Feature Selection Info\n")
        f.write(f"--------------------------------------------------\n")
        f.write(f"Target Column:        {stats['target_col']}\n")
        f.write(f"Importance Threshold: {stats['threshold']}\n")
        f.write(f"Original Features:    {stats['n_original']}\n")
        f.write(f"Selected Features:    {stats['n_selected']}\n")
        f.write(f"Dropped Features:     {stats['n_dropped']} (Low Importance)\n")
        f.write(f"Drop Rate:            {stats['drop_rate']:.2f}%\n\n")
        f.write(f"[2] Resource & Performance\n")
        f.write(f"--------------------------------------------------\n")
        f.write(f"Total Execution Time: {stats['total_time']:.2f} sec ({stats['total_time']/60:.2f} min)\n")
        f.write(f"Peak GPU Memory Used: {stats['peak_memory']:.2f} GB\n")
        f.write(f"Training Epochs:      {stats['epochs']}\n")
        f.write(f"Generated Samples:    {stats['n_generated']}\n")
    
    logging.info(f"[{dataset_name}] 實驗報告已儲存至: {report_path}")

def run_experiment(dataset_name):
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    meta = DATASET_META.get(dataset_name)
    if not meta: return

    # 設定 Input 路徑 (讀取 real_train 用)
    dataset_input_dir = os.path.join(ROOT_DIR, dataset_name)
    train_path = os.path.join(dataset_input_dir, "real_train.csv")
    
    # 設定 Output 路徑 (存檔用) - 會自動建立 .../filtered_experiments/dataset_name/
    dataset_output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        logging.error(f"[{dataset_name}] 'real_train.csv' not found at {train_path}. Skipping.")
        return

    logging.info(f"[{dataset_name}] Loading real_train.csv...")
    try:
        train_data = pd.read_csv(train_path)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to read csv: {e}")
        return

    # 1. 計算特徵重要性
    target_col = meta["target"]
    logging.info(f"[{dataset_name}] Analyzing Feature Importance (Target: {target_col})...")
    
    if target_col not in train_data.columns:
        logging.error(f"[{dataset_name}] Target '{target_col}' not found.")
        return

    all_weights = FeatureImportanceAnalyzer.get_importance(train_data, target_col)
    
    # 2. 特徵篩選
    logging.info(f"[{dataset_name}] Filtering features (Threshold >= {IMPORTANCE_THRESHOLD})...")
    
    selected_features = [
        feat for feat, score in all_weights.items() 
        if score >= IMPORTANCE_THRESHOLD or feat == target_col
    ]
    filtered_weights = {k: all_weights[k] for k in selected_features}
    
    n_original = len(train_data.columns)
    n_selected = len(selected_features)
    n_dropped = n_original - n_selected
    logging.info(f"[{dataset_name}] Dropped {n_dropped} low-importance features.")
    
    train_data_filtered = train_data[selected_features]
    
    # [修改] 存檔權重表到新的 Output 目錄
    try:
        weight_df = pd.DataFrame(list(filtered_weights.items()), columns=['Feature', 'Importance'])
        weight_df = weight_df.sort_values(by='Importance', ascending=False)
        weight_df.to_csv(os.path.join(dataset_output_dir, "feature_weights_filtered.csv"), index=False)
    except: pass

    # 3. 初始化與訓練
    # [修改] Checkpoint 存到新的 Output 目錄
    checkpoint_dir = os.path.join(dataset_output_dir, "checkpoint")

    model = WeightedGReaT(
        llm='distilgpt2',       
        batch_size=8,           
        epochs=100,             
        experiment_dir=checkpoint_dir
    )

    logging.info(f"[{dataset_name}] Start Filtered & Weighted Fine-Tuning...")
    
    model.fit(
        train_data_filtered, 
        feature_weights=filtered_weights, 
        hard_sorting=True,  # 這裡維持硬排序
        conditional_col=target_col,
        random_conditional_col=False
    )
    
    logging.info(f"[{dataset_name}] Training Completed.")
    
    # 4. 生成
    n_synth = len(train_data)
    logging.info(f"[{dataset_name}] Generating {n_synth} samples...")
    
    synthetic_data = model.sample(
        n_samples=n_synth,
        temperature=0.7,    
        device="cuda",      
        max_length=512      
    )
    
    # [修改] 合成資料存到新的 Output 目錄
    synth_path = os.path.join(dataset_output_dir, "synthetic.csv")
    synthetic_data.to_csv(synth_path, index=False)
    logging.info(f"[{dataset_name}] Saved to {synth_path}")
    
    # 5. 結算與報告
    end_time = time.time()
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    stats = {
        'target_col': target_col,
        'threshold': IMPORTANCE_THRESHOLD,
        'n_original': n_original,
        'n_selected': n_selected,
        'n_dropped': n_dropped,
        'drop_rate': (n_dropped / n_original) * 100,
        'total_time': end_time - start_time,
        'peak_memory': peak_memory,
        'epochs': 100,
        'n_generated': n_synth
    }
    # [修改] 報告存到新的 Output 目錄
    save_experiment_report(dataset_output_dir, dataset_name, stats)
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    target_datasets = [
        #"breast_cancer", 
        #"heart_cleveland", 
        #"pima_diabetes", 
        #"mammographic_mass", 
        #"ilpd", 

        "heart_failure_clinical_records", 
        "german_credit", # 測試這個修正
        "parkinsons",    # 以及這個
        "heart_statlog", 
        "australian_credit"
    ]

    print(f"Starting Feature Selection Experiment (Threshold={IMPORTANCE_THRESHOLD})...")
    print(f"Input Dir:  {ROOT_DIR}")
    print(f"Output Dir: {OUTPUT_ROOT}")
    
    for ds_name in target_datasets:
        try:
            run_experiment(ds_name)
        except Exception as e:
            logging.error(f"Critical Failure in {ds_name}: {str(e)}")
            continue