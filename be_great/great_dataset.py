import random
import typing as tp
import numpy as np

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """GReaT Dataset
    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.
    """

    def set_tokenizer(self, tokenizer, float_precision=None):
        self.tokenizer = tokenizer
        self.float_precision = float_precision
        self.feature_weights = None
        self.is_hard_sorting = False  # [New] 預設為關閉 (軟排序)

    def set_feature_weights(self, weights: dict):
        """Set importance weights for each feature."""
        self.feature_weights = weights

    def set_hard_sorting(self, enabled: bool):
        """[New] Enable or disable hard sorting (deterministic order)."""
        self.is_hard_sorting = enabled

    def _format_value(self, value):
        if isinstance(value, (float, np.floating)) and self.float_precision is not None:
            formatted_value_str = f"{value:.{self.float_precision}f}"
            if '.' in formatted_value_str:
                formatted_value_str = formatted_value_str.rstrip('0').rstrip('.')
            return formatted_value_str
        return str(value).strip()

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        row = self._data.fast_slice(key, 1)
        col_names = row.column_names
        num_cols = row.num_columns

        # --- 排列邏輯核心 ---
        if self.feature_weights:
            # 1. 準備權重陣列
            weights = np.array([self.feature_weights.get(name, 1.0) for name in col_names])
            
            if self.is_hard_sorting:
                # [模式 A: 硬排序]
                # 直接依權重由大到小排序 (argsort 預設由小到大，所以加負號變成由大到小)
                # 這會產生固定的順序：Target -> Most Important -> ... -> Least Important
                shuffle_idx = np.argsort(-weights).tolist()
            else:
                # [模式 B: 軟排序 (加權隨機)]
                weight_sum = weights.sum()
                probs = weights / weight_sum if weight_sum > 0 else np.ones(num_cols) / num_cols
                shuffle_idx = np.random.choice(range(num_cols), size=num_cols, replace=False, p=probs).tolist()
        else:
            # [模式 C: 純隨機 (原版)]
            shuffle_idx = list(range(num_cols))
            random.shuffle(shuffle_idx)
        # --------------------

        shuffled_text = ", ".join(
            [
                "%s is %s"
                % (row.column_names[i], self._format_value(row.columns[i].to_pylist()[0]))
                for i in shuffle_idx
            ]
        )
        tokenized_text = self.tokenizer(shuffled_text, padding=True)
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch