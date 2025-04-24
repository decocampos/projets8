import pandas as pd
from PyQt5.QtWidgets import QFileDialog

def load_data(parent=None):
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(parent, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)", options=options)
    if file_path:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type")
        return df, file_path
    return None, None
