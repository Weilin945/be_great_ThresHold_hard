# filename: run_generation.py
import os
import logging
import time
import datetime
import pandas as pd
import numpy as np
import torch
from custom_great import WeightedGReaT, FeatureImportanceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 1. 路徑與參數設定
# ==========================================
# 原始資料路徑 (Input)
ROOT_DIR = "/home/panda3/Research/CT/be_great_Threshold_hard/Data"

# 設定輸出路徑 (Output) - 與 Data 同層級
# 會自動建立 .../filtered_experiments 資料夾
OUTPUT_ROOT = os.path.join(os.path.dirname(ROOT_DIR), "filtered_experiments")

# 設定刪除比例 (20%)
DROP_RATE = 0.20 

# 訓練參數 (用於計算每個 Epoch 時間)
EPOCHS = 100
BATCH_SIZE = 8


DATASET_META = {
    "breast_cancer": {"target": "target"},
    "heart_cleveland": {"target": "target"},
    "pima_diabetes": {"target": "target"},
    "mammographic_mass": {"target": "target"},
    "ilpd": {"target": "target"},
    "heart_failure_clinical_records": {"target": "target"},
    "parkinsons": {"target": "target"},
    "heart_statlog": {"target": "target"},
    "german_credit": {"target": "target"},
    "australian_credit": {"target": "target"}
}

def get_gpu_memory_mb():
    """獲取當前記錄到的 GPU 峰值記憶體 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def save_experiment_report(output_dir, dataset_name, stats):
    """將實驗數據寫入人類可讀的 Efficiency Report"""
    report_path = os.path.join(output_dir, "efficiency_report.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== GTT Integrated Efficiency Report: {dataset_name} ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"==================================================\n\n")
        
        f.write(f"[1] Feature Selection (Drop Bottom {int(stats['drop_setting']*100)}%)\n")
        f.write(f"  - Original Features:    {stats['n_original']}\n")
        f.write(f"  - Selected Features:    {stats['n_selected']}\n")
        f.write(f"  - Dropped Count:        {stats['n_dropped']}\n")
        f.write(f"  - Weight Report:        Saved to 'feature_weights_filtered.csv'\n\n")
        
        f.write(f"[2] Training Efficiency\n")
        f.write(f"  - Total Train Time:     {stats['train_time']:.2f} sec\n")
        f.write(f"  - Time per Epoch:       {stats['time_per_epoch']:.4f} sec\n")
        f.write(f"  - Max GPU Memory:       {stats['train_gpu_mem']:.2f} MB\n\n")
        
        f.write(f"[3] Sampling Efficiency\n")
        f.write(f"  - Generated Samples:    {stats['n_generated']}\n")
        f.write(f"  - Total Sample Time:    {stats['sample_time']:.2f} sec\n")
        f.write(f"  - Throughput:           {stats['throughput']:.2f} rows/sec\n")
        f.write(f"  - Max GPU Memory:       {stats['sample_gpu_mem']:.2f} MB\n\n")
        
        f.write(f"[4] Overall\n")
        f.write(f"  - Total Runtime:        {stats['total_runtime']:.2f} sec\n")
        f.write(f"==================================================\n")
    
    logging.info(f"[{dataset_name}] Efficiency Report saved to: {report_path}")

def run_experiment(dataset_name):
    # 總計時開始
    global_start_time = time.time()
    
    meta = DATASET_META.get(dataset_name)
    if not meta: return

    # 設定 Input/Output 路徑
    dataset_input_dir = os.path.join(ROOT_DIR, dataset_name)
    train_path = os.path.join(dataset_input_dir, "real_train.csv")
    dataset_output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        logging.error(f"[{dataset_name}] 'real_train.csv' not found. Skipping.")
        return

    # 1. 讀取與計算權重
    logging.info(f"[{dataset_name}] Loading & Analyzing...")
    try:
        train_data = pd.read_csv(train_path)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to read csv: {e}")
        return

    target_col = meta["target"]
    if target_col not in train_data.columns:
        logging.error(f"[{dataset_name}] Target '{target_col}' not found.")
        return

    all_weights = FeatureImportanceAnalyzer.get_importance(train_data, target_col)
    
    # 2. 特徵篩選 (Drop Bottom 20%)
    logging.info(f"[{dataset_name}] Filtering features (Dropping bottom {DROP_RATE*100}%)...")
    sorted_features = sorted(all_weights.items(), key=lambda x: x[1], reverse=True)
    n_total = len(sorted_features)
    n_drop = int(n_total * DROP_RATE)
    n_keep = n_total - n_drop
    
    selected_features = [feat for feat, score in sorted_features[:n_keep]]
    filtered_weights = {k: all_weights[k] for k in selected_features}
    
    train_data_filtered = train_data[selected_features]
    
    # 存檔權重表
    pd.DataFrame(list(filtered_weights.items()), columns=['Feature', 'Importance'])\
      .sort_values(by='Importance', ascending=False)\
      .to_csv(os.path.join(dataset_output_dir, "feature_weights_filtered.csv"), index=False)

    # 3. 初始化模型
    checkpoint_dir = os.path.join(dataset_output_dir, "checkpoint")
    model = WeightedGReaT(
        llm='distilgpt2',       
        batch_size=BATCH_SIZE,           
        epochs=EPOCHS,             
        experiment_dir=checkpoint_dir
    )

    # ==========================
    # Phase A: Training
    # ==========================
    logging.info(f"[{dataset_name}] Start Training...")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    
    model.fit(
        train_data_filtered, 
        feature_weights=filtered_weights, 
        hard_sorting=True,
        conditional_col=target_col,
        random_conditional_col=False
    )
    
    train_end = time.time()
    train_time = train_end - train_start
    train_gpu_mem = get_gpu_memory_mb()
    logging.info(f"[{dataset_name}] Training Done. Time: {train_time:.2f}s")

    # ==========================
    # Phase B: Sampling
    # ==========================
    n_synth = len(train_data)
    logging.info(f"[{dataset_name}] Start Sampling ({n_synth} rows)...")
    
    # 重置 GPU 統計以便單獨計算 Sampling 的消耗
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    sample_start = time.time()
    
    synthetic_data = model.sample(
        n_samples=n_synth,
        temperature=0.7,    
        device="cuda",      
        max_length=512      
    )
    
    sample_end = time.time()
    sample_time = sample_end - sample_start
    sample_gpu_mem = get_gpu_memory_mb()
    
    # 存檔合成資料
    synth_path = os.path.join(dataset_output_dir, "synthetic.csv")
    synthetic_data.to_csv(synth_path, index=False)
    logging.info(f"[{dataset_name}] Sampling Done. Time: {sample_time:.2f}s")
    
    # ==========================
    # Phase C: Reporting
    # ==========================
    global_end_time = time.time()
    
    stats = {
        'drop_setting': DROP_RATE,
        'n_original': n_total,
        'n_selected': n_keep,
        'n_dropped': n_drop,
        
        # Training Stats
        'train_time': train_time,
        'time_per_epoch': train_time / EPOCHS if EPOCHS > 0 else 0,
        'train_gpu_mem': train_gpu_mem,
        
        # Sampling Stats
        'n_generated': n_synth,
        'sample_time': sample_time,
        'throughput': n_synth / sample_time if sample_time > 0 else 0,
        'sample_gpu_mem': sample_gpu_mem,
        
        # Overall
        'total_runtime': global_end_time - global_start_time
    }
    
    save_experiment_report(dataset_output_dir, dataset_name, stats)
    
    # 清理資源
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
        "german_credit", 
        "parkinsons",    
        "heart_statlog", 
        "australian_credit"
    ]

    print(f"Starting Efficiency Experiment (Drop Bottom {int(DROP_RATE*100)}%)...")
    print(f"Input Dir:  {ROOT_DIR}")
    print(f"Output Dir: {OUTPUT_ROOT}")
    
    for ds_name in target_datasets:
        try:
            run_experiment(ds_name)
        except Exception as e:
            logging.error(f"Critical Failure in {ds_name}: {str(e)}")
            continue