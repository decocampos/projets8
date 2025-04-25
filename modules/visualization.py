import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

def plot_histogram(df, column):
    plt.figure(figsize=(6,4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()

def plot_elbow_kmeans(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(7,4))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (Within-cluster SS)')
    plt.title('Elbow Method for KMeans Clustering')
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.show()

def plot_clusters_kmeans(df, labels):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('KMeans Clusters (2D PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def plot_boxplot(df, column):
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

def plot_elbow_kmeans(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(7,4))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (Within-cluster SS)')
    plt.title('Elbow Method for KMeans Clustering')
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.show()

def plot_clusters_kmeans(df, labels):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('KMeans Clusters (2D PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def plot_heatmap(df):
    # Only use numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        plt.figure(figsize=(6,2))
        plt.text(0.5, 0.5, "Not enough numeric columns for correlation heatmap", ha='center', va='center')
        plt.axis('off')
        plt.show()
        return
    corr = numeric_df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_countplot(df, column):
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x=column)
    plt.title(f"Countplot of {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, selected_cols):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.pairplot(df[selected_cols].dropna())
    plt.suptitle(f"Pairplot: {', '.join(selected_cols)}", y=1.02)
    plt.show()

def plot_jointplot(df, col1, col2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    g = sns.jointplot(data=df, x=col1, y=col2, kind='scatter')
    g.fig.suptitle(f'Jointplot: {col1} vs {col2}', y=1.02)
    plt.show()

def plot_regression(df, x_col, y_col, regression_result):
    import matplotlib.pyplot as plt
    import numpy as np
    score, reg = regression_result if isinstance(regression_result, tuple) else (regression_result, None)
    X = df[[x_col]].dropna()
    y = df[y_col].dropna()
    # align X and y
    df_xy = df[[x_col, y_col]].dropna()
    X = df_xy[x_col]
    y = df_xy[y_col]
    plt.figure(figsize=(7,5))
    plt.scatter(X, y, color='blue', label='Data')
    if reg is not None:
        y_pred = reg.predict(X.values.reshape(-1,1))
        plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Regression: {y_col} ~ {x_col}\nR² = {score:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_elbow_kmeans(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(7,4))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (Within-cluster SS)')
    plt.title('Elbow Method for KMeans Clustering')
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.show()

def plot_clusters_kmeans(df, labels):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('KMeans Clusters (2D PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.show()


def plot_elbow_kmeans(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(7,4))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (Within-cluster SS)')
    plt.title('Elbow Method for KMeans Clustering')
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.show()

def plot_clusters_kmeans(df, labels):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('KMeans Clusters (2D PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def plot_categorical_by_target(df, target_col):
    cat_cols = [col for col in df.select_dtypes(include=['category']).columns if col != target_col]
    n = len(cat_cols)
    if n == 0:
        print("No other categorical variables to plot.")
        return

    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        sns.countplot(data=df, x=col, hue=target_col, ax=ax)
        ax.set_title(f"{col} by {target_col}")
        ax.legend(title=target_col)
        ax.tick_params(axis='x', rotation=45)

        # Crosstab para a tabela
        ct = pd.crosstab(df[col], df[target_col])

        # Posiciona a tabela mais abaixo
        table = ax.table(
            cellText=ct.values,
            rowLabels=ct.index,
            colLabels=ct.columns,
            cellLoc='center',
            bbox=[0, -0.45, 1, 0.35]  # y=-0.45 empurra ainda mais para baixo; ajuste a altura (0.35) se quiser
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        ax.set_xlabel(col)

    # Esconde eixos sobrando
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    # Espaçamento extra entre linhas e margem inferior
    fig.subplots_adjust(hspace=0.6, bottom=0.25)
    plt.show()