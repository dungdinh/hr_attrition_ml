import pandas as pd

from src.utils import infer_target_column, normalize_target


def test_normalize_target_yes_no():
    y = pd.Series(["Yes", "No", " yes ", "NO"])
    out = normalize_target(y)
    assert out.tolist() == [1, 0, 1, 0]


def test_normalize_target_numeric_strings():
    y = pd.Series(["1", "0", 1, 0])
    out = normalize_target(y)
    assert out.tolist() == [1, 0, 1, 0]


def test_infer_target_column():
    df = pd.DataFrame({"Attrition": ["Yes", "No"], "Age": [30, 40]})
    assert infer_target_column(df) == "Attrition"
