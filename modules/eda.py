import pandas as pd
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt

def get_summary(df):
    return df.describe(include='all').transpose()

def get_missing_values(df):
    return df.isnull().sum()

def impute_missing_values(df, column):
    """Impute missing values in a column using mean for numeric and mode for categorical"""
    if df[column].dtype.name in ['float64', 'int64']:
        # For numeric columns, use mean
        imputed_value = df[column].mean()
    else:
        # For categorical columns, use mode
        imputed_value = df[column].mode()[0]
    
    df[column] = df[column].fillna(imputed_value)
    return df

class MissingValuesDialog(QDialog):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle("Missing Values")
        self.resize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create table for missing values
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Column", "Missing Values"])
        
        # Get missing values
        missing = self.df.isnull().sum()
        self.table.setRowCount(len(missing))
        
        row = 0
        for column, count in missing.items():
            self.table.setItem(row, 0, QTableWidgetItem(str(column)))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))
            row += 1
            
        layout.addWidget(self.table)
        
        # Add impute button
        button_layout = QHBoxLayout()
        self.impute_btn = QPushButton("Impute Selected Column")
        self.impute_btn.clicked.connect(self.impute_selected)
        button_layout.addWidget(self.impute_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def impute_selected(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            column = self.table.item(current_row, 0).text()
            self.df = impute_missing_values(self.df, column)
            # Update the missing values count
            missing_count = self.df[column].isnull().sum()
            self.table.setItem(current_row, 1, QTableWidgetItem(str(missing_count)))
