import pandas as pd
import sys
import pytest
import numpy as np
sys.path.append("../src")
from data_processing import optimize_memory

def test_optimize_memory():
    df = pd.DataFrame({
        "age": [25.0, 30.0, 35.0],
        "weight": [70.0, 80.0, 90.0],
        "count": [1, 2, 3]
    })
    
    df_optimized = optimize_memory(df)
    
    assert df_optimized["age"].dtype == "float32"
    assert df_optimized["weight"].dtype == "float32"
    assert df_optimized["count"].dtype == "int32"
    
    print(" optimize_memory test passed!")

test_optimize_memory()

from src.data_processing import load_data   # adjust import if needed

def test_load_data(tmp_path):
    # 1. Create a small sample DataFrame
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'NObeyesdad': ['normal', 'overweight', 'obese']
    })
    
    # 2. Save it to a temporary CSV file
    csv_file = tmp_path / "sample.csv"
    sample_data.to_csv(csv_file, index=False)
    
    # 3. Call the function with the temporary file
    X, y = load_data(filepath=str(csv_file), target_col="NObeyesdad")
    
    # 4. Assertions
    # Check X: should be a DataFrame without the target column
    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == ['feature1', 'feature2']
    assert X.shape == (3, 2)
    assert X['feature1'].tolist() == [1, 2, 3]
    
    # Check y: should be a numpy array (raveled) with the target values
    assert isinstance(y, np.ndarray)
    assert y.tolist() == ['normal', 'overweight', 'obese']
    assert y.shape == (3,)  # 1D array
    
    # Optional: check that index was reset
    assert X.index.tolist() == [0, 1, 2]