import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QMessageBox, QInputDialog, QComboBox, QPushButton, QDialog, QLabel, QHBoxLayout, QListWidget
from PyQt5.QtCore import Qt
import pandas as pd
from modules import data_loader, eda, visualization, mining

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyDataMiner")
        self.resize(900, 600)
        self.df = None
        self.init_ui()

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        load_action = QAction('Load Data', self)
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)

        # Visualization menu
        vis_menu = menubar.addMenu('Visualization')
        hist_action = QAction('Histogram', self)
        hist_action.triggered.connect(self.show_histogram)
        vis_menu.addAction(hist_action)
        box_action = QAction('Boxplot', self)
        box_action.triggered.connect(self.show_boxplot)
        vis_menu.addAction(box_action)
        heat_action = QAction('Correlation Heatmap', self)
        heat_action.triggered.connect(self.show_heatmap)
        vis_menu.addAction(heat_action)
        # New: Categorical by Target
        cat_by_target_action = QAction('Categorical by Target', self)
        cat_by_target_action.triggered.connect(self.show_categorical_by_target)
        vis_menu.addAction(cat_by_target_action)

        # Data Mining menu
        mining_menu = menubar.addMenu('Data Mining')
        classify_action = QAction('Classification (Random Forest)', self)
        classify_action.triggered.connect(self.run_classification)
        mining_menu.addAction(classify_action)
        regress_action = QAction('Regression (Linear)', self)
        regress_action.triggered.connect(self.run_regression)
        mining_menu.addAction(regress_action)
        cluster_action = QAction('Clustering (KMeans)', self)
        cluster_action.triggered.connect(self.run_clustering)
        mining_menu.addAction(cluster_action)

        # EDA menu
        eda_menu = menubar.addMenu('EDA')
        summary_action = QAction('Summary Statistics', self)
        summary_action.triggered.connect(self.show_summary)
        eda_menu.addAction(summary_action)
        missing_action = QAction('Missing Values', self)
        missing_action.triggered.connect(self.show_missing)
        eda_menu.addAction(missing_action)

        self.table = QTableWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.table)

        # Add buttons for column deletion and value insertion
        button_layout = QHBoxLayout()
        self.delete_col_btn = QPushButton("Delete Column")
        self.delete_col_btn.clicked.connect(self.delete_column)
        self.insert_val_btn = QPushButton("Insert Value")
        self.insert_val_btn.clicked.connect(self.insert_value)
        button_layout.addWidget(self.delete_col_btn)
        button_layout.addWidget(self.insert_val_btn)
        layout.addLayout(button_layout)

        self.update_buttons_state()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_data(self):
        df, path = data_loader.load_data(self)
        if df is not None:
            # Convert object columns to category
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')
            self.df = df
            self.show_data(df)
            self.setWindowTitle(f"PyDataMiner - {path}")
        else:
            QMessageBox.warning(self, "Load Data", "No file loaded or unsupported format.")
        self.update_buttons_state()

    def show_data(self, df):
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.table.clear()
        if df is not None and not df.empty:
            self.table.setColumnCount(len(df.columns))
            self.table.setHorizontalHeaderLabels(df.columns)
            for i, row in df.iterrows():
                self.table.insertRow(i)
                for j, val in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(str(val)))
                if i > 99:
                    break  # Display only first 100 rows for performance
        self.update_buttons_state()

    def show_histogram(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        # Only allow numeric columns
        numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        if not numeric_cols:
            QMessageBox.warning(self, "No Numeric Columns", "No numeric columns available for histogram.")
            return
        col, ok = QInputDialog.getItem(self, "Select Column", "Column:", numeric_cols, 0, False)
        if ok:
            visualization.plot_histogram(self.df, col)

    def show_boxplot(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        col, ok = QInputDialog.getItem(self, "Select Column", "Column:", list(self.df.columns), 0, False)
        if ok:
            visualization.plot_boxplot(self.df, col)

    def show_heatmap(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        visualization.plot_heatmap(self.df)

    def show_categorical_by_target(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        # Only allow categorical columns
        cat_cols = [col for col in self.df.select_dtypes(include=['category']).columns]
        if not cat_cols:
            QMessageBox.warning(self, "No Categorical", "No categorical columns available.")
            return
        target_col, ok = QInputDialog.getItem(self, "Select Target", "Target variable:", cat_cols, 0, False)
        if ok and target_col:
            try:
                visualization.plot_categorical_by_target(self.df, target_col)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def show_summary(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        summary = eda.get_summary(self.df)
        dlg = QDialog(self)
        dlg.setWindowTitle("Summary Statistics")
        layout = QVBoxLayout()
        label = QLabel(summary.to_string())
        label.setTextInteractionFlags(label.textInteractionFlags() | Qt.TextSelectableByMouse)
        layout.addWidget(label)
        dlg.setLayout(layout)
        dlg.resize(700, 400)
        dlg.exec_()

    def show_missing(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        missing = eda.get_missing_values(self.df)
        dlg = QDialog(self)
        dlg.setWindowTitle("Missing Values")
        layout = QVBoxLayout()
        label = QLabel(missing.to_string())
        label.setTextInteractionFlags(label.textInteractionFlags() | Qt.TextSelectableByMouse)
        layout.addWidget(label)
        dlg.setLayout(layout)
        dlg.resize(500, 300)
        dlg.exec_()

    def run_classification(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        col, ok = QInputDialog.getItem(self, "Target Column", "Select target column:", list(self.df.columns), 0, False)
        if ok:
            try:
                result = mining.classify_rf(self.df, col)
                dlg = QDialog(self)
                dlg.setWindowTitle("Classification Results")
                layout = QVBoxLayout()
                label = QLabel(f"Accuracy: {result['accuracy']:.3f}\n\nConfusion Matrix:\n{result['confusion_matrix']}\n\nReport:\n{result['report']}")
                label.setTextInteractionFlags(label.textInteractionFlags() | Qt.TextSelectableByMouse)
                layout.addWidget(label)
                dlg.setLayout(layout)
                dlg.resize(700, 400)
                dlg.exec_()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def run_regression(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        col, ok = QInputDialog.getItem(self, "Target Column", "Select target column:", list(self.df.columns), 0, False)
        if ok:
            try:
                score = mining.regress_linear(self.df, col)
                QMessageBox.information(self, "Regression R^2", f"R^2 Score: {score:.3f}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def run_clustering(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        # Show elbow plot for K selection
        try:
            visualization.plot_elbow_kmeans(self.df)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in elbow plot: {e}")
            return
        n, ok = QInputDialog.getInt(self, "Clusters", "Number of clusters:", 3, 2, 10, 1)
        if ok:
            try:
                labels = mining.cluster_kmeans(self.df, n)
                visualization.plot_clusters_kmeans(self.df, labels)
                QMessageBox.information(self, "Clustering", f"Cluster labels assigned to {len(labels)} rows.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def update_buttons_state(self):
        has_data = self.df is not None and not self.df.empty
        self.delete_col_btn.setEnabled(has_data)
        self.insert_val_btn.setEnabled(not has_data)

    def delete_column(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "No data to delete columns from.")
            return
        dlg = MultiColumnSelectDialog(self, list(self.df.columns))
        if dlg.exec_() == QDialog.Accepted:
            selected_cols = dlg.get_selected_columns()
            if selected_cols:
                self.df.drop(columns=selected_cols, inplace=True)
                self.show_data(self.df)

    def insert_value(self):
        if self.df is not None and not self.df.empty:
            QMessageBox.warning(self, "Data Exists", "Insert is only allowed when there is no data.")
            return
        col, ok1 = QInputDialog.getText(self, "Insert Value", "Enter column name:")
        if not ok1 or not col:
            return
        val, ok2 = QInputDialog.getText(self, "Insert Value", f"Enter value for column '{col}':")
        if not ok2:
            return
        # Create a new DataFrame with the value
        self.df = pd.DataFrame({col: [val]})
        self.show_data(self.df)

class MultiColumnSelectDialog(QDialog):
    def __init__(self, parent, columns):
        super().__init__(parent)
        self.setWindowTitle("Delete Columns")
        self.selected_columns = []
        layout = QVBoxLayout()
        label = QLabel("Select columns to delete:")
        layout.addWidget(label)
        self.list_widget = QListWidget()
        self.list_widget.addItems(columns)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.list_widget)
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.resize(300, 400)

    def get_selected_columns(self):
        return [item.text() for item in self.list_widget.selectedItems()]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
