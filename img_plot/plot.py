import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager as fm

def plot_model_performance_with_annotations():
    model_names = [
        'XGBoost',
        'LightGBM',
        'CatBoost',
        'SimpleNN',
        'MergeNN',
        'Random Forest',
    ]
    model_performance = {
        'XGBoost': {
            'None, None': 0.834567,
            'Std, One-Hot': 0.810903,
        },
        'LightGBM': {
            'None, None': 0.828752,
            'Std, One-Hot': 0.831177,
        },
        'CatBoost': {
            'None, None': 0.866916,
            'Std, One-Hot': 0.858284,
        },
        'SimpleNN': {
            'None, None': 0.843949,
            'Std, One-Hot': 0.843499,
        },
        'MergeNN': {
            'None, None': 0.870199,
            'Std, One-Hot': 0.861821,
        },
        'Random Forest': {
            'None, None': 0.672808,
            'Std, One-Hot': 0.677940,
        },
    }

    preprocess_methods = ['(None, None)', '(Std, One-Hot)']
    bar_width = 0.35  
    x = np.arange(len(model_names)) 

    none_none_values = [model_performance[model]['None, None'] for model in model_names]
    std_one_hot_values = [model_performance[model]['Std, One-Hot'] for model in model_names]

    # Plot the data
    plt.figure(figsize=(14, 7))
    bars1 = plt.bar(x - bar_width / 2, none_none_values, bar_width, label='None, None', color='skyblue')
    bars2 = plt.bar(x + bar_width / 2, std_one_hot_values, bar_width, label='Std, One-Hot', color='salmon')

    # Add annotations above each bar
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{bar.get_height():.4f}',
                 ha='center', va='bottom', fontsize=10, color='blue')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{bar.get_height():.4f}',
                 ha='center', va='bottom', fontsize=10, color='red')

    # Add labels and title
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Performance (AUC)', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim((0.5, 1))
    plt.legend(title='Preprocess(Numeric, Categoric)')
    plt.tight_layout()

    # Show the plot
    plt.show()
    return 

def plot_ensemble_model_performance():

    # Data
    methods = ['None, None', 'None, Normalization', 'None, One-Hot', 
            'Std, None', 'Std, Normalization', 'Std, One-Hot', 
            'Normalization, None', 'Normalization, Normalization', 'Normalization, One-Hot']
    scores = [0.870199, 0.869414, 0.866306, 0.865755, 0.865099, 0.861821, 0.771376, 0.770122, 0.764972]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(methods, scores, color='skyblue')

    # Add labels and title
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.xlabel('Preprocessing Methods(Numeric, Categoric)', fontsize=12)
    plt.ylabel('Performance (AUC)', fontsize=12)
    plt.title('Performance for Different Preprocessing Methods in MergeNN', fontsize=14)

    # Display values on top of bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.002, f'{score:.4f}', ha='center', fontsize=10)

    # Show plot
    plt.ylim((0.6, 0.9))
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()
    return

def plot_catboost_performance():

    methods = ['with \"cat_features\"', "without \"cat_features\""]
    scores = [0.857915, 0.866916]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(methods, scores, color='skyblue')

    # Add labels and title
    plt.xticks()
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Performance (AUC)', fontsize=12)
    plt.title('CatBoost with/without cat_features parameter', fontsize=14)

    # Display values on top of bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.002, f'{score:.4f}', ha='center', fontsize=10)

    # Show plot
    plt.ylim((0.75, 0.9))
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()
    return

def plot_model_performance_v2():
# Data
    methods = ['xgboost', 'catboost', 'ensemble(xgboost + catboost + lightGBM)']
    scores = [0.838274, 0.836335, 0.861708]

    # Plot
    plt.figure(figsize=(14, 7))
    plt.bar(methods, scores, color='skyblue')

    # Add labels and title
    plt.xticks()
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Performance (AUC)', fontsize=12)
    plt.title('Performance for Different Models', fontsize=14)

    # Display values on top of bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.002, f'{score:.4f}', ha='center', fontsize=10)

    # Show plot
    plt.ylim((0.6, 0.9))
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()
    return

def plot_percentage_of_workload():

    font_path = os.path.join(os.path.dirname(__file__), 'KAIU.TTF')
    prop = fm.FontProperties(fname = font_path)

    teammates = [
        '江曉明',
        '陳柏淮',
        '楊承翰',
    ]
    programming = [
        40,
        20,
        40,
    ]
    report = [
        60, 
        40, 
    ]

    fig, ax = plt.subplots(figsize = (5, 5))
    wedges, texts, autotexts = ax.pie(programming, labels=teammates, autopct='%1.1f%%', startangle = 140, colors = plt.cm.Paired.colors)

    for text in texts:
        text.set_fontproperties(prop)
        text.set_fontsize(15)

    ax.set_title('程式撰寫', fontproperties=prop, fontsize = 20)


    fig2, ax2 = plt.subplots(figsize = (5, 5))
    wedges, texts, autotexts = ax2.pie(report, labels=teammates[:len(report)], autopct='%1.1f%%', startangle = 140, colors = plt.cm.Paired.colors)

    for text in texts:
        text.set_fontproperties(prop)
        text.set_fontsize(15)

    ax2.set_title('報告撰寫', fontproperties=prop, fontsize = 20)

    plt.show()

    return 

if __name__ == '__main__':
    # plot_model_performance_with_annotations()
    # plot_ensemble_model_performance()
    # plot_catboost_performance()
    # plot_percentage_of_workload()
    plot_model_performance_v2()