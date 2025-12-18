# filename: run_hard_sort.py
import os
import logging
import pandas as pd
import numpy as np
from custom_great import WeightedGReaT, FeatureImportanceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = "/home/panda3/Research/CT/be_great_hard/experiments"

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

def run_experiment(dataset_name):
    meta = DATASET_META.get(dataset_name)
    if not meta: return

    dataset_dir = os.path.join(ROOT_DIR, dataset_name)
    train_path = os.path.join(dataset_dir, "real_train.csv")
    
    if not os.path.exists(train_path):
        logging.error(f"[{dataset_name}] 'real_train.csv' not found. Skipping.")
        return

    logging.info(f"[{dataset_name}] Loading real_train.csv...")
    try:
        train_data = pd.read_csv(train_path)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to read csv: {e}")
        return

    target_col = meta["target"]
    logging.info(f"[{dataset_name}] Analyzing Feature Importance (Target: {target_col})...")
    
    if target_col not in train_data.columns:
        logging.error(f"[{dataset_name}] Target '{target_col}' not found.")
        return

    weights = FeatureImportanceAnalyzer.get_importance(train_data, target_col)
    
    # 建立硬排序專用資料夾
    hard_sort_dir = os.path.join(dataset_dir, "hard_sort_experiment")
    os.makedirs(hard_sort_dir, exist_ok=True)
    checkpoint_dir = os.path.join(hard_sort_dir, "checkpoint")

    model = WeightedGReaT(
        llm='distilgpt2',       
        batch_size=8,           
        epochs=100,             
        experiment_dir=checkpoint_dir
    )

    logging.info(f"[{dataset_name}] Start HARD SORT Fine-Tuning...")
    
    # [關鍵修正] 
    # 1. conditional_col=target_col: 指定採樣起點為 Target
    # 2. random_conditional_col=False: 禁止隨機切換起點
    # 3. hard_sorting=True: 開啟硬排序
    model.fit(
        train_data, 
        feature_weights=weights, 
        hard_sorting=True,
        conditional_col=target_col,     # [Fix 1]
        random_conditional_col=False    # [Fix 2]
    )
    
    logging.info(f"[{dataset_name}] Training Completed.")
    
    n_synth = len(train_data)
    logging.info(f"[{dataset_name}] Generating {n_synth} samples (Starting from {target_col})...")
    
    # 因為在 fit 時已經鎖定了 conditional_col 為 target_col，
    # 這裡 sample() 會自動使用 Target 作為 Prompt，符合模型學到的順序。
    synthetic_data = model.sample(
        n_samples=n_synth,
        temperature=0.7,    
        device="cuda",      
        max_length=512      
    )
    
    synth_path = os.path.join(hard_sort_dir, "synthetic.csv")
    synthetic_data.to_csv(synth_path, index=False)
    
    logging.info(f"[{dataset_name}] Hard-sort synthetic data saved to {synth_path}")
    
    import torch
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 這裡放你要跑的資料集
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

    print(f"Starting HARD SORT Experiment on {len(target_datasets)} Datasets...")
    
    for ds_name in target_datasets:
        try:
            run_experiment(ds_name)
        except Exception as e:
            logging.error(f"Critical Failure in {ds_name}: {str(e)}")
            continue