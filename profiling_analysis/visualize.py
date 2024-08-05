import pandas as pd
import numpy as np
import json
import traceback, sys
import matplotlib.pyplot as plt
import seaborn as sns
from profiling_analysis import logger


def prepare_data(df, threshold, time_percentage_col, operation_col):
    try:
        small_slices = df[df[time_percentage_col] < threshold]
        if not small_slices.empty:
            other_data = pd.DataFrame({time_percentage_col: [small_slices[time_percentage_col].sum()], 
                                    operation_col: ['Others']})
            df = pd.concat([df[df[time_percentage_col] >= threshold], other_data], ignore_index=True)
        return df
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def get_others_items(df, threshold, time_percentage_col, operation_col):
    try:
        small_slices = df[df[time_percentage_col] < threshold]
        if not small_slices.empty:
            other_items = small_slices[operation_col].tolist()
            return other_items
        else:
            return []
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    

def plot_pie_chart(df, save_path, title, threshold, percentage_col, stat_col, stat_base, operation_col, colors, legend_title, plt_title, figsize=(10, 10)):
    try:
        # Prepare the data for the pie chart
        df_filtered = prepare_data(df, threshold, percentage_col, operation_col)
        
        # Get the items that are grouped as others
        others_items = get_others_items(df, threshold, percentage_col, operation_col)
        
        # Setup pie chart plot
        plt.figure(figsize=figsize)  # Increase figure size for better readability

        # Plot pie chart without showing percentages on the pie
        # wedges, texts, autotexts = plt.pie(df_filtered[percentage_col],
        #                                    startangle=140, colors=colors, textprops={'fontsize': 12}, autopct='%1.1f%%')
        # wedges, texts = plt.pie(df_filtered[percentage_col], startangle=140, colors=colors, textprops={'fontsize': 12})
        wedges, texts = plt.pie(df_filtered[percentage_col], startangle=140, colors=colors, textprops={'fontsize': 14})

        # Calculate the percentages for the pie chart
        total = sum(df_filtered[percentage_col])
        percentages = [f'{(x/total) * 100:.3f}%' for x in df_filtered[percentage_col]]

        # Round time values to four decimal places and add suffix
        df_filtered["Stat_Col"] = df_filtered[stat_col].apply(lambda x: f'{x:.2f} {stat_base}')

        # Each operation must not be more than 50 characters
        df_filtered[operation_col] = df_filtered[operation_col].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        
        # Create custom labels for each slice with times included
        legend_labels = [f'{op} - {perc} ({stat})' for op, perc, stat in zip(df_filtered[operation_col], percentages, df_filtered["Stat_Col"])]
        
        # Adding a legend to handle small slices or clarify the chart
        plt.legend(wedges, legend_labels, title=legend_title, loc="best", fontsize='small')

        # Better title display
        # plt.title(plt_title, fontsize=14, fontweight='bold')
        plt.title(plt_title, fontsize=16, fontweight='bold')
        
        # Show plot with adjustments
        # plt.show()
        plt.savefig(save_path)
        
        return others_items
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    

def plot_bar_chart(df, save_path, title, threshold, percentage_col, stat_col, stat_base, operation_col, x_label, y_label, bar_label, bar_color='b', figsize=(8, 4), legends=True, bar_width = 0.35, xlabel_rotation=25):
    try:
        # Prepare the data for the bar chart
        df_bar = df.sort_values(by=percentage_col, ascending=False)
        
        # Set "Others" for items less than threshold
        df_bar = prepare_data(df_bar, threshold, percentage_col, operation_col)
        
        # Get the list of items that are aggregated into 'Others'
        other_items = get_others_items(df, threshold, percentage_col, operation_col)
        
        # Setup bar chart plot
        plt.figure(figsize=figsize)  # Increase figure size for better readability
        
        # Plot bar chart with additional information
        fig, ax = plt.subplots(figsize=figsize)
        
        index = np.arange(len(df_bar))
        
        # Each operation must not be more than 50 characters
        df_bar[operation_col] = df_bar[operation_col].apply(lambda x: x[:50])
        
        # Plot bars
        bar1 = ax.bar(index, df_bar[percentage_col], bar_width, label=bar_label, color=bar_color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(index)
        ax.set_xticklabels(df_bar[operation_col], rotation=xlabel_rotation)
        
        if legends:
            ax.legend()
            
        # Add data labels
        for i, v in enumerate(df_bar[percentage_col]):
            ax.text(i, v + 0.5, str(round(v, 2)), color='black', ha='center')
            # ax.text(i, v + 0.01, f'{v:.4f}', color='black', ha='center')
            
        plt.grid(True)  # Add grid for better visibility
        plt.tight_layout()  # Adjust layout to not cut off labels
        # plt.show()
        plt.savefig(save_path)
        
        return other_items
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    

def plot_time(df, save_path, title, limit, time_col, name_col, label_col, x_label, y_label, legend_label, bar_color='b', figsize=(12, 6), legends = False, bar_width = 0.35, xlabel_rotation=45):
    try:
        # Prepare the data for the bar chart
        df_sorted = df.sort_values(by=time_col, ascending=False).head(limit)
        
        # Setup bar chart plot
        plt.figure(figsize=figsize)  # Increase figure size for better readability
        
        bar_width = 0.35  # Increase bar width for better visibility
        index = np.arange(len(df_sorted))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bars
        bars = ax.bar(index, df_sorted[time_col], bar_width, label=legend_label, color=bar_color)
        
        # ax.set_xlabel(x_label, fontsize=14)
        # ax.set_ylabel(y_label, fontsize=14)
        # ax.set_title(title, fontsize=16)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(title, fontsize=18)
        
        ax.set_xticks(index)
        # ax.set_xticklabels(df_sorted[name_col], rotation=xlabel_rotation, fontsize=12)  # Use operation names as labels
        ax.set_xticklabels(df_sorted[name_col], rotation=xlabel_rotation, fontsize=14)  # Use operation names as labels
        
        if legends:
            # ax.legend(fontsize=12)
            ax.legend(fontsize=14)
        
        # Add data labels from label_col
        for bar in bars:
            yval = bar.get_height()
            # ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02*max(df_sorted[time_col]), f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02*max(df_sorted[time_col]), f'{yval:.2f}', ha='center', va='bottom', fontsize=14)
            
        plt.grid(True)  # Add grid
        plt.tight_layout()  # Adjust layout to not cut off labels
        # plt.show()
        plt.savefig(save_path)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e