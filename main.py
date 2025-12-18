import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from transformers import TrainingArguments

# 引用 GReaT 套件
from be_great import GReaT
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer
from be_great.great.py import RandomConditionalColumnCallback # 假設這個 Callback 定義在 great.py 中，若無法引用可直接複製類別定義過來

# -----------------------------------------------------------
# [Step 1] 特徵重要性計算 (完全保留你的邏輯)
# -----------------------------------------------------------
def get_feature_importance(df, target_col):
    """
    計算特徵重要性，回傳權重字典。
    """
    print(f"正在分析特徵重要性 (Target: {target_col})...")
    df_encoded = df.copy()
    
    # 簡單預處理
    enc = OrdinalEncoder()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        df_encoded[cat_cols] = enc.fit_transform(df_encoded[cat_cols].astype(str))
    
    # 填補缺失值
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # 模型選擇
    if df[target_col].nunique() < 20 and df[target_col].dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model.fit(X, y)
    
    # 正規化權重
    importances = model.feature_importances_
    feature_names = X.columns
    weights = dict(zip(feature_names, importances))
    weights[target_col] = 1.0  # 強制設定 Target 為最高權重
    
    return weights

# -----------------------------------------------------------
# [Step 2 & 3] 繼承 GReaT 並注入權重邏輯
# -----------------------------------------------------------
class MyGReaT(GReaT):
    """
    自定義的 GReaT 類別，支援 feature_weights 參數。
    """
    def fit(
        self,
        data,
        column_names=None,
        conditional_col=None,
        resume_from_checkpoint=False,
        random_conditional_col=True,
        # [新增參數]
        feature_weights=None 
    ):
        # 1. 轉換數據格式
        df = self._array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # 2. 建立 Dataset (這裡是關鍵)
        # 我們使用修改過的 GReaTDataset (假設你已經修改了 great_dataset.py)
        print("轉換數據為 HuggingFace Dataset...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, self.float_precision)
        
        # [關鍵注入] 如果有傳入權重，就設定進去
        if feature_weights:
            print(f"[System] 偵測到特徵權重，正在注入 Dataset...")
            # 注意：這裡前提是你已經在 GReaTDataset 中實作了 set_feature_weights 方法
            if hasattr(great_ds, 'set_feature_weights'):
                great_ds.set_feature_weights(feature_weights)
            else:
                print("[Warning] GReaTDataset 尚未實作 set_feature_weights，權重將被忽略！")

        # 3. 設定訓練參數 (從原始 fit 方法複製過來)
        print("建立 Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )
        
        # 設定 Callbacks
        callbacks = []
        if random_conditional_col:
            # 這裡需要引用 RandomConditionalColumnCallback，如果 import 失敗，
            # 可以直接在這裡重新定義該 class 或者暫時註解掉
            try:
                from be_great.great import RandomConditionalColumnCallback
                callbacks.append(RandomConditionalColumnCallback(self, df))
            except ImportError:
                pass # 忽略或手動定義
        
        # 4. 初始化 Trainer
        great_trainer = GReaTTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
            callbacks=callbacks,
        )

        # 5. 開始訓練
        print("開始訓練 (Training Start)...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

# -----------------------------------------------------------
# [Main] 執行區塊
# -----------------------------------------------------------
if __name__ == "__main__":
    # 1. 讀取數據
    # 假設你有一個 csv 檔案
    # df = pd.read_csv("your_data.csv")
    
    # (這裡用假數據模擬，讓你直接能跑)
    df = pd.DataFrame({
        'Age': np.random.randint(20, 60, 100),
        'Income': np.random.randint(30000, 80000, 100),
        'Education': np.random.choice(['HighSchool', 'Bachelor', 'Master'], 100),
        'Target': np.random.choice([0, 1], 100)
    })
    TARGET_COLUMN = "Target"

    # 2. 計算權重
    weights = get_feature_importance(df, TARGET_COLUMN)
    print("計算出的權重:", weights)

    # 3. 初始化自定義模型
    model = MyGReaT(llm='distilgpt2', batch_size=8, epochs=1)

    # 4. 開始訓練 (傳入 feature_weights)
    model.fit(df, feature_weights=weights)
    
    print("訓練完成！模型現在學會了基於重要性的特徵排序。")