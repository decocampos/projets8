# 📊 PyDataMiner

> A powerful, user-friendly desktop application for data analysis and visualization, built with Python and PyQt5.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt-5-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 📥 Data Import/Export
- Load CSV and Excel files
- Export processed data to CSV or Excel

### 🔍 Exploratory Data Analysis
- View summary statistics
- Detect and handle missing values
  - Smart imputation (mean for numeric, mode for categorical)
  - Interactive missing values dialog
- Discretize variables

### 📈 Visualization
- 📊 Histograms for numeric variables
- 📦 Boxplots for distribution analysis
- 🌡️ Correlation heatmaps
- 📉 Categorical plots
- 📊 Count plots for categorical data
- 🔄 Pairplots for numeric variables
- 🎯 Joint plots for relationship analysis

### 🤖 Data Mining
- 🌳 Classification with Random Forest
- 📈 Linear Regression
- 🎯 K-Means Clustering

## 🚀 Quick Start

### Option 1: Download Executable
1. Go to the [Releases](../../releases) page
2. Download the latest `PyDataMiner.exe`
3. Double-click to run!

### Option 2: Run from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PyDataMiner.git
cd PyDataMiner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## 🛠️ Technical Stack

- **GUI**: PyQt5
- **Data Processing**: pandas
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn

## 📖 Usage Guide

1. **Loading Data**
   - Click `File > Load Data`
   - Select your CSV or Excel file

2. **Exploring Data**
   - Use `EDA` menu for:
     - Summary statistics
     - Missing value analysis
     - Variable discretization

3. **Visualizing Data**
   - Access all plots from `Visualization` menu
   - Select variables when prompted
   - Plots open in separate windows

4. **Data Mining**
   - Choose from `Data Mining` menu:
     - Classification
     - Regression
     - Clustering
   - Follow the prompts to select variables

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
2. Run the application:
   ```bash
   python main.py
   ```

## Structure
- `main.py`: Entry point, launches GUI
- `modules/`: Data loading, EDA, visualization, mining logic
- `ui/`: UI resources
