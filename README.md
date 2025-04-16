# PyDataMiner

A modular, user-friendly desktop application for exploratory data analysis and data mining, inspired by Weka and built with Python libraries (pandas, matplotlib, seaborn, scikit-learn, PyQt5).

## Features
- Import datasets (CSV, Excel)
- Data exploration and cleaning
- Visualization (histograms, boxplots, heatmaps, etc.)
- Data mining: classification, regression, clustering
- Export results and figures

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python main.py
   ```

## Structure
- `main.py`: Entry point, launches GUI
- `modules/`: Data loading, EDA, visualization, mining logic
- `ui/`: UI resources
