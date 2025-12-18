import os
import numpy as np
import pandas as pd
import warnings
from scipy.stats import entropy
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ==========================================
# 1. 實驗參數與路徑設定
# ==========================================
# 真實資料路徑 (Real Data)
ROOT_DIR = "/home/panda3/Research/CT/be_great_Threshold_hard/experiments"

# 合成資料路徑 (Synthetic Data)
FILTERED_ROOT = "/home/panda3/Research/CT/be_great_Threshold_hard/filtered_experiments"

# 指定資料集
TARGET_DATASETS = [
    "breast_cancer", 
    "heart_cleveland", 
    "pima_diabetes", 
    "mammographic_mass", 
    "ilpd",

    #"heart_failure_clinical_records",
    #"german_credit", 
    #"parkinsons", 
    #"heart_statlog", 
    #"australian_credit"
]

# 針對特定資料集指定 Target 欄位名稱
DATASET_PARAMS = {
    "parkinsons": {"target": "status"}, 
    "heart_statlog": {"target": "heart-disease"},
    "australian_credit": {"target": "A15"}
}

SEEDS = [42, 100, 123] 

MODELS = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'XGB': XGBClassifier(use_label_encoder=False, random_state=42)
}

# ==========================================
# 2. 核心前處理器
# ==========================================
def get_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

# ==========================================
# 3. 指標計算工具
# ==========================================
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    auc = 0.5
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            n_classes = len(np.unique(y_test))
            
            if n_classes == 2:
                if y_prob.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except:
        pass

    return acc, f1, auc

def compute_inverse_kl(df_real, df_syn):
    kl_scores = []
    common_cols = [c for c in df_real.columns if c in df_syn.columns]
    
    for col in common_cols:
        try:
            real_data = pd.to_numeric(df_real[col], errors='coerce').fillna(0)
            syn_data = pd.to_numeric(df_syn[col], errors='coerce').fillna(0)
            
            min_val = min(real_data.min(), syn_data.min())
            max_val = max(real_data.max(), syn_data.max())
            
            if min_val == max_val:
                bins = 10
            else:
                bins = np.linspace(min_val, max_val, 20)
            
            p, _ = np.histogram(real_data, bins=bins, density=True)
            q, _ = np.histogram(syn_data, bins=bins, density=True)
            
            p = np.where(p == 0, 1e-10, p)
            q = np.where(q == 0, 1e-10, q)
            
            kl = entropy(p, q)
            kl_scores.append(kl)
        except:
            continue
            
    avg_kl = np.mean(kl_scores) if kl_scores else 10.0
    return 1 / (1 + avg_kl)

def compute_discriminator_score(X_real, X_syn):
    X = np.vstack([X_real, X_syn])
    y = np.hstack([np.ones(len(X_real)), np.zeros(len(X_syn))])
    
    if len(X) < 10: return 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    return score

def compute_density_coverage(X_real, X_syn, k=5):
    k = min(k, len(X_real)-1)
    if k < 1: return 0.0, 0.0

    nbrs_real = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_real)
    real_to_real_dist, _ = nbrs_real.kneighbors(X_real)
    radii = real_to_real_dist[:, -1]
    
    syn_to_real_dist, syn_to_real_idx = nbrs_real.kneighbors(X_syn)
    
    covered_real_samples = np.zeros(len(X_real), dtype=bool)
    density_count = 0
    
    for i in range(len(X_syn)):
        nearest_real_idx = syn_to_real_idx[i, 0]
        dist = syn_to_real_dist[i, 0]
        if dist <= radii[nearest_real_idx]:
            covered_real_samples[nearest_real_idx] = True
            density_count += 1
            
    coverage = np.mean(covered_real_samples)
    density = density_count / len(X_syn)
    
    return density, coverage

# ==========================================
# 4. Report Generator
# ==========================================
def save_human_readable_report(dataset_name, q_metrics, mle_results, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"=================================================\n")
        f.write(f"   Evaluation Report: {dataset_name}\n")
        f.write(f"=================================================\n\n")
        
        f.write(f"[1] Quality Metrics\n")
        f.write(f"-------------------------------------------------\n")
        f.write(f"  * Discriminator Score: {q_metrics['Disc_Score']:.4f} (Ideal: 0.5)\n")
        f.write(f"  * Inverse KL:          {q_metrics['Inv_KL']:.4f}\n")
        f.write(f"  * Density:             {q_metrics['Density']:.4f}\n")
        f.write(f"  * Coverage:            {q_metrics['Coverage']:.4f}\n\n")
        
        f.write(f"[2] Downstream MLE Results\n")
        f.write(f"-------------------------------------------------\n")
        f.write(f"{'Model':<10} | {'TRTR F1':<12} | {'TSTR F1':<12} | {'Gap':<12} | {'Augmented F1':<12} | {'Aug Gain':<10}\n")
        f.write(f"-"*80 + "\n")
        
        for res in mle_results:
            model = res['Model']
            trtr = f"{res['TRTR_F1_Mean']:.4f}"
            tstr = f"{res['TSTR_F1_Mean']:.4f}"
            gap = f"{res['F1_Gap']:.4f}"
            aug = f"{res['Augmented_F1_Mean']:.4f}"
            gain = f"{res['F1_Aug_Gain']:.4f}"
            
            f.write(f"{model:<10} | {trtr:<12} | {tstr:<12} | {gap:<12} | {aug:<12} | {gain:<10}\n")

# ==========================================
# 5. Main Experiment Loop (Fixed v7: Path & Intersection)
# ==========================================
def run_experiment(dataset_folder):
    dataset_name = os.path.basename(dataset_folder)
    print(f"\nProcessing Dataset: {dataset_name}...")
    
    # [修正] 修改讀取路徑邏輯
    # 優先從 FILTERED_ROOT (平行資料夾) 讀取 synthetic.csv
    syn_path = os.path.join(FILTERED_ROOT, dataset_name, "synthetic.csv")
    
    # 如果找不到，印出錯誤
    if not os.path.exists(syn_path):
        print(f"   [Error] Synthetic data not found at: {syn_path}")
        # 這裡不自動退回 local，因為你的結構已經很明確分開了，避免讀錯檔案
        return None, None
        
    try:
        real_train = pd.read_csv(os.path.join(dataset_folder, "real_train.csv"))
        real_test = pd.read_csv(os.path.join(dataset_folder, "real_test.csv"))
        synthetic = pd.read_csv(syn_path)
        print(f"   [Info] Loaded synthetic data from: {syn_path}")
    except Exception as e:
        print(f"   [Error] Skipping {dataset_name}: {e}")
        return None, None

    # Target Column Logic
    if dataset_name in DATASET_PARAMS:
        target_col = DATASET_PARAMS[dataset_name]["target"]
    else:
        target_col = real_train.columns[-1]

    if target_col not in synthetic.columns:
         print(f"   [Error] Target '{target_col}' missing in synthetic data.")
         return None, None

    # ========================================================
    # [核心修正] 自動取交集欄位 (Intersection Alignment)
    # ========================================================
    # 找出 Real 和 Synthetic 共有的特徵 (排除 Target)
    real_cols = set(real_train.columns)
    syn_cols = set(synthetic.columns)
    
    common_features = list(real_cols.intersection(syn_cols))
    if target_col in common_features:
        common_features.remove(target_col)
    
    # 排序以確保一致性
    common_features.sort()
    
    # 如果有特徵被丟棄，印出警告提示
    dropped_by_filter = len(real_cols) - len(common_features) - 1 # -1 是因為 target
    if dropped_by_filter > 0:
        print(f"   [Info] Aligning features: {len(real_train.columns)-1} -> {len(common_features)}")
        print(f"   [Info] Dropped {dropped_by_filter} features during evaluation alignment.")

    # 重新對齊所有資料集 (只保留交集特徵 + Target)
    final_cols = common_features + [target_col]
    
    real_train = real_train[final_cols]
    real_test = real_test[final_cols]
    synthetic = synthetic[final_cols]
    # ========================================================
    
    X_real_train = real_train.drop(columns=[target_col])
    y_real_train = real_train[target_col]
    X_real_test = real_test.drop(columns=[target_col])
    y_real_test = real_test[target_col]
    X_syn = synthetic.drop(columns=[target_col])
    y_syn = synthetic[target_col]

    # === Type Alignment ===
    if pd.api.types.is_numeric_dtype(y_real_train):
        try:
            y_real_train = y_real_train.astype(int)
            y_real_test = y_real_test.astype(int)
            y_syn = y_syn.astype(float).round().astype(int)
        except Exception as e:
            print(f"   [Warning] Target casting failed: {e}")
    
    # === Label Encoding ===
    le = LabelEncoder()
    all_labels = set(y_real_train.astype(str)) | set(y_real_test.astype(str)) | set(y_syn.astype(str))
    le.fit(list(all_labels))
    n_classes = len(le.classes_) 
    
    y_real_train_enc = le.transform(y_real_train.astype(str))
    y_real_test_enc = le.transform(y_real_test.astype(str))
    y_syn_enc = le.transform(y_syn.astype(str))
    
    # === Feature Preprocessing ===
    # 注意：這裡已經是用「瘦身後」的 X_real_train 來 fit，所以不會報錯
    preprocessor = get_preprocessor(X_real_train)
    preprocessor.fit(X_real_train)
    
    X_rt_enc = preprocessor.transform(X_real_train)
    X_test_enc = preprocessor.transform(X_real_test)
    X_syn_enc = preprocessor.transform(X_syn)

    # === Part A: Quality Metrics ===
    print("   -> Calculating Quality Metrics...")
    disc_score = compute_discriminator_score(X_rt_enc, X_syn_enc)
    inv_kl = compute_inverse_kl(real_train, synthetic)
    density, coverage = compute_density_coverage(X_rt_enc, X_syn_enc)
    
    quality_metrics = {
        'Dataset': dataset_name,
        'Disc_Score': disc_score,
        'Inv_KL': inv_kl,
        'Density': density,
        'Coverage': coverage
    }

    # === Part B: MLE Scenarios ===
    print("   -> Running MLE Scenarios...")
    mle_results = []
    
    X_aug = np.vstack([X_rt_enc, X_syn_enc])
    y_aug = np.hstack([y_real_train_enc, y_syn_enc])

    scenarios = {
        'TRTR': (X_rt_enc, y_real_train_enc),
        'TSTR': (X_syn_enc, y_syn_enc),
        'Augmented': (X_aug, y_aug)
    }
    
    for model_name, model_base in MODELS.items():
        seed_scores = {scen: {'acc': [], 'f1': [], 'auc': []} for scen in scenarios}
        
        for seed in SEEDS:
            model = clone(model_base)
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)
            
            if model_name == 'XGB':
                if n_classes > 2:
                    model.set_params(objective='multi:softmax', eval_metric='mlogloss', num_class=n_classes)
                else:
                    model.set_params(objective='binary:logistic', eval_metric='logloss')

            for scen_name, (X_train_s, y_train_s) in scenarios.items():
                try:
                    model.fit(X_train_s, y_train_s)
                    acc, f1, auc = calculate_metrics(model, X_test_enc, y_real_test_enc)
                    seed_scores[scen_name]['acc'].append(acc)
                    seed_scores[scen_name]['f1'].append(f1)
                    seed_scores[scen_name]['auc'].append(auc)
                except Exception as e:
                    # print(f"      [Warning] Model {model_name} failed on {scen_name}: {e}")
                    seed_scores[scen_name]['acc'].append(0)
                    seed_scores[scen_name]['f1'].append(0)
                    seed_scores[scen_name]['auc'].append(0)
        
        row_res = {'Dataset': dataset_name, 'Model': model_name}
        for scen_name in scenarios:
            for metric in ['acc', 'f1', 'auc']:
                vals = seed_scores[scen_name][metric]
                if vals:
                    row_res[f'{scen_name}_{metric.upper()}_Mean'] = np.mean(vals)
                    row_res[f'{scen_name}_{metric.upper()}_Std'] = np.std(vals)
                else:
                    row_res[f'{scen_name}_{metric.upper()}_Mean'] = 0.0
                    row_res[f'{scen_name}_{metric.upper()}_Std'] = 0.0
        
        row_res['F1_Gap'] = row_res['TRTR_F1_Mean'] - row_res['TSTR_F1_Mean']
        row_res['F1_Aug_Gain'] = row_res['Augmented_F1_Mean'] - row_res['TRTR_F1_Mean']
        mle_results.append(row_res)
        
    return quality_metrics, mle_results

def main():
    print(f"Starting Evaluation on {len(TARGET_DATASETS)} datasets...\n")
    print(f"Real Data Dir: {ROOT_DIR}")
    print(f"Synthetic Data Dir: {FILTERED_ROOT}")
    
    summary_table_list = []
    
    for ds_name in TARGET_DATASETS:
        dataset_path = os.path.join(ROOT_DIR, ds_name)
        
        if not os.path.exists(dataset_path):
            print(f"[Warning] Folder not found: {dataset_path}")
            continue
            
        q_metrics, mle_res = run_experiment(dataset_path)
        
        if q_metrics and mle_res:
            df_mle_ds = pd.DataFrame(mle_res)
            summary_row = {
                'Dataset': ds_name,
                'Avg_TRTR_F1': df_mle_ds['TRTR_F1_Mean'].mean(),
                'Avg_TSTR_F1': df_mle_ds['TSTR_F1_Mean'].mean(),
                'Avg_Augmented_F1': df_mle_ds['Augmented_F1_Mean'].mean(),
                'Avg_F1_Gap': df_mle_ds['F1_Gap'].mean(),
                'Avg_Aug_Gain': df_mle_ds['F1_Aug_Gain'].mean(),
                'Disc_Score': q_metrics['Disc_Score'],
                'Inv_KL': q_metrics['Inv_KL'],
                'Density': q_metrics['Density'],
                'Coverage': q_metrics['Coverage']
            }
            summary_table_list.append(summary_row)
            
            # 即時存檔
            txt_report_path = os.path.join(dataset_path, "Eval_Report_Readable.txt")
            save_human_readable_report(ds_name, q_metrics, mle_res, txt_report_path)
            
            df_mle_ds.to_csv(os.path.join(dataset_path, "Eval_Detailed_MLE.csv"), index=False)
            pd.DataFrame([summary_row]).to_csv(os.path.join(dataset_path, "Eval_Summary.csv"), index=False)
            
            print(f"   [Saved] Reports saved to {dataset_path}")

    if summary_table_list:
        df_summary = pd.DataFrame(summary_table_list)
        cols_order = ['Dataset', 'Avg_TRTR_F1', 'Avg_TSTR_F1', 'Avg_Augmented_F1', 
                      'Avg_F1_Gap', 'Avg_Aug_Gain', 
                      'Disc_Score', 'Inv_KL', 'Density', 'Coverage']
        df_summary = df_summary[cols_order]
        
        print("\n=== Final Summary Table ===")
        print(df_summary.to_string(index=False))
        df_summary.to_csv("Final_Summary_Table.csv", index=False)
        print("\n[Success] Final_Summary_Table.csv saved.")
    else:
        print("\n[Error] No results generated.")

if __name__ == "__main__":
    main()