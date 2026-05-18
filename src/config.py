from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hr_attrition.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
PLOT_DIR = BASE_DIR / "plots"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Attrition"
POSITIVE_LABEL = "Yes"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 2
N_ITER_SEARCH = 2

LEAKAGE_COLUMNS = {
    "Attrition Reason",
    "AttritionDate",
    "IsVoluntary",
    "Employee Name",
    "Manager Name",
    "Join Date",
}
ID_COLUMNS = {
    "Employee ID",
    "EmployeeNumber",
}
