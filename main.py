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

        # Save As submenu
        save_menu = file_menu.addMenu('Save As')
        save_csv_action = QAction('Save as CSV', self)
        save_csv_action.triggered.connect(self.save_as_csv)
        save_menu.addAction(save_csv_action)
        save_excel_action = QAction('Save as Excel', self)
        save_excel_action.triggered.connect(self.save_as_excel)
        save_menu.addAction(save_excel_action)

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
        cat_by_target_action = QAction('Categorical by Target', self)
        cat_by_target_action.triggered.connect(self.show_categorical_by_target)
        vis_menu.addAction(cat_by_target_action)

        # Countplot for categorical variable
        countplot_action = QAction('Countplot (Categorical)', self)
        countplot_action.triggered.connect(self.show_countplot)
        vis_menu.addAction(countplot_action)

        # Pairplot and Jointplot
        pairplot_action = QAction('Pairplot (Numeric)', self)
        pairplot_action.triggered.connect(self.show_pairplot)
        vis_menu.addAction(pairplot_action)
        jointplot_action = QAction('Jointplot (Numeric)', self)
        jointplot_action.triggered.connect(self.show_jointplot)
        vis_menu.addAction(jointplot_action)

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
        discretize_action = QAction('Discretize Variable', self)
        discretize_action.triggered.connect(self.show_discretize_variable)
        eda_menu.addAction(discretize_action)

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

    def show_countplot(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        # Only allow categorical columns
        cat_cols = [col for col in self.df.select_dtypes(include=['category']).columns]
        if not cat_cols:
            QMessageBox.warning(self, "No Categorical", "No categorical columns available.")
            return
        col, ok = QInputDialog.getItem(self, "Select Column", "Categorical column:", cat_cols, 0, False)
        if ok and col:
            try:
                visualization.plot_countplot(self.df, col)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def save_as_csv(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save as CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.df.to_csv(path, index=False)
                QMessageBox.information(self, "Save CSV", f"File saved: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def save_as_excel(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save as Excel", "", "Excel Files (*.xlsx *.xls)")
        if path:
            try:
                self.df.to_excel(path, index=False)
                QMessageBox.information(self, "Save Excel", f"File saved: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

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
        numeric_cols = [col for col in self.df.select_dtypes(include=['number']).columns]
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Not Enough Numeric Columns", "Need at least 2 numeric columns for regression.")
            return
        x_col, ok1 = QInputDialog.getItem(self, "Select X Variable", "Select independent variable (X):", numeric_cols, 0, False)
        if not ok1:
            return
        y_col, ok2 = QInputDialog.getItem(self, "Select Y Variable", "Select dependent variable (Y):", [col for col in numeric_cols if col != x_col], 0, False)
        if not ok2:
            return
        try:
            import modules.visualization as visualization
            score = mining.regress_linear(self.df, y_col, x_col)
            visualization.plot_regression(self.df, x_col, y_col, score)
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
        dlg.setWindowTitle("Select Multiple Variables")
        if dlg.exec_() == QDialog.Accepted:
            selected_cols = dlg.get_selected_columns()
            if selected_cols:
                reply = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete column(s): {', '.join(selected_cols)}?", QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.df.drop(columns=selected_cols, inplace=True)
                    self.show_data(self.df)

    def show_pairplot(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        numeric_cols = [col for col in self.df.select_dtypes(include=['number']).columns]
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Not Enough Numeric Columns", "Need at least 2 numeric columns for pairplot.")
            return
        # Multi-selection dialog
        dlg = MultiColumnSelectDialog(self, numeric_cols)
        dlg.setWindowTitle("Select Multiple Variables")
        if dlg.exec_() == QDialog.Accepted:
            selected_cols = dlg.get_selected_columns()
            if len(selected_cols) < 2:
                QMessageBox.warning(self, "Select Variables", "Please select at least 2 numeric variables.")
                return
            import modules.visualization as visualization
            visualization.plot_pairplot(self.df, selected_cols)

    def show_jointplot(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        numeric_cols = [col for col in self.df.select_dtypes(include=['number']).columns]
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Not Enough Numeric Columns", "Need at least 2 numeric columns for jointplot.")
            return
        col1, ok1 = QInputDialog.getItem(self, "Select X Variable", "X variable:", numeric_cols, 0, False)
        if not ok1:
            return
        col2, ok2 = QInputDialog.getItem(self, "Select Y Variable", "Y variable:", numeric_cols, 0, False)
        if not ok2 or col1 == col2:
            return
        import modules.visualization as visualization
        visualization.plot_jointplot(self.df, col1, col2)

    def show_discretize_variable(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        numeric_cols = [col for col in self.df.select_dtypes(include=['number']).columns]
        if not numeric_cols:
            QMessageBox.warning(self, "No Numeric Columns", "No numeric columns available for discretization.")
            return
        col, ok = QInputDialog.getItem(self, "Select Variable", "Numeric variable:", numeric_cols, 0, False)
        if not ok:
            return
        bins, ok2 = QInputDialog.getInt(self, "Number of bins", "How many bins?", 4, 2, 20, 1)
        if not ok2:
            return
        import pandas as pd
        new_col = f"{col}_bin"
        try:
            self.df[new_col] = pd.cut(self.df[col], bins)
            self.show_data(self.df)
            QMessageBox.information(self, "Discretization", f"Column '{new_col}' added.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

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
        label = QLabel("Select columns:")
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
