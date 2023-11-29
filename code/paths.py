from pathlib import Path


DATASETS_DIR = Path("/home/shsingh/knowledge_distillation/dataset/scratch")
MICCAI_BraTS2020_Data = DATASETS_DIR / "MICCAI_BraTS2020_Data"
DATAFRAME_DIR = DATASETS_DIR / 'dataframes'

MODEL_DIR = Path("kd_models")
OUTPUT_DIR = Path('kfold_result')