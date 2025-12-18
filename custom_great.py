# filename: custom_great.py
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from transformers import TrainingArguments

# 引用 GReaT 原始套件
from be_great import GReaT
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import _array_to_dataframe

# 嘗試引用 Callback
try:
    from be_great.great import RandomConditionalColumnCallback
except ImportError:
    RandomConditionalColumnCallback = None

class FeatureImportanceAnalyzer:
    """智囊團：負責分析資料並計算特徵重要性"""
    
    @staticmethod
    def get_importance(df: pd.DataFrame, target_col: str) -> dict:
        logging.info(f"   [Analyzer] Calculating feature importance for target: '{target_col}'")
        df_encoded = df.copy()
        
        # 1. 簡單預處理
        enc = OrdinalEncoder()
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df_encoded[cat_cols] = enc.fit_transform(df_encoded[cat_cols].astype(str))
        
        df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
        for col in cat_cols:
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])

        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]

        # 2. 模型選擇
        is_classification = (df[target_col].nunique() < 50) or (df[target_col].dtype == 'object')
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
        model.fit(X, y)
        
        # 3. 權重計算
        importances = model.feature_importances_
        feature_names = X.columns
        weights = dict(zip(feature_names, importances))
        
        # 強制設定 Target 為最高權重
        weights[target_col] = 1.0 
        
        top_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:4]
        logging.info(f"   [Analyzer] Top features: {top_features}")
        
        return weights

class WeightedGReaT(GReaT):
    """執行者：支援加權排序與硬排序的 GReaT 模型"""
    
    def fit(self, data, column_names=None, conditional_col=None, resume_from_checkpoint=False, 
            random_conditional_col=True, feature_weights=None, hard_sorting=False):
        """
        [修改重點] 覆蓋 fit 方法，加入 hard_sorting 參數
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        logging.info("   [WeightedGReaT] Creating dataset...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, self.float_precision)
        
        # 設定權重與排序模式
        if feature_weights:
            logging.info("   [WeightedGReaT] Injecting feature importance weights...")
            if hasattr(great_ds, 'set_feature_weights'):
                great_ds.set_feature_weights(feature_weights)
                
                # [新增邏輯] 設定硬排序
                if hard_sorting:
                    logging.info("   [WeightedGReaT] !!! Enabling HARD SORTING (Deterministic Order) !!!")
                    # 呼叫我們剛剛在 great_dataset.py 新增的方法
                    if hasattr(great_ds, 'set_hard_sorting'):
                        great_ds.set_hard_sorting(True)
                else:
                    logging.info("   [WeightedGReaT] Using Soft Weighted Permutation (Default)")
            else:
                logging.warning("   [Warning] GReaTDataset missing 'set_feature_weights'. Using random order.")

        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )
        
        callbacks = []
        if random_conditional_col and RandomConditionalColumnCallback:
            callbacks.append(RandomConditionalColumnCallback(self, df))
        
        great_trainer = GReaTTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
            callbacks=callbacks,
        )

        logging.info("   [WeightedGReaT] Starting training loop...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer