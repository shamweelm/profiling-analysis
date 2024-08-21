import pandas as pd
import numpy as np
import json
import traceback, sys
import matplotlib.pyplot as plt
from profiling_analysis.configs.constants import INFERENCE_NVTX_REPORTS_PATH, INFERENCE_OPERATIONS_PATH, INFERENCE_OPERATIONS_MAPPING
from profiling_analysis.visualize import plot_bar_chart, plot_pie_chart
import seaborn as sns
import ast
import re
from datetime import datetime
from profiling_analysis.helpers import convert_string_to_dataframe, get_operation_mapping, get_reports_path, get_unique_kernel_names_inference, get_visualization_path, map_kernel_name
from profiling_analysis import logger

def plot_cuda_api_summary(task, base_path, category, name, data, title, threshold=1.0, time_percentage_col='Time (%)', operation_col='Operation', figsize=(6, 6)):
    try:
        # Prepare data
        # data = json.loads(data.values[0])
        data = data.values[0]
        
        # Convert to DataFrame
        df = convert_string_to_dataframe(data)
        
        save_path_pie_chart = get_visualization_path(
                base_path, f"{task}_cuda_api_summary_pie_chart.png", operation=f"{name}/cuda_api", category = category
            )
        others_items_pie_chart = plot_pie_chart(df = df,
                    save_path = save_path_pie_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Avg (ns)',
                    stat_base = 'ns',
                    legend_title = operation_col,
                    operation_col = operation_col,
                    colors = sns.color_palette("tab20", len(df)),
                    plt_title = title,
                    figsize=figsize)
        
        save_path_bar_chart = get_visualization_path(
                base_path, f"{task}_cuda_api_summary_bar_chart.png", operation=f"{name}/cuda_api", category = category
            )
        other_items_bar_chart = plot_bar_chart(df = df,
                    save_path=save_path_bar_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Time (s)',
                    stat_base = 's',
                    operation_col = operation_col,
                    x_label = operation_col,
                    y_label = time_percentage_col,
                    bar_label = time_percentage_col,
                    bar_color = 'b',
                    figsize = figsize,
                    legends = False,
                    bar_width = 0.35,
                    xlabel_rotation = 0)
        
        # Save other items to a file
        others_items = {
            "others_items_pie_chart": others_items_pie_chart,
            "other_items_bar_chart": other_items_bar_chart,
        }

        other_items_file_path = get_visualization_path(
            base_path, f"{task}_cuda_api_summary_other_items.csv", operation=f"{name}/cuda_api", category = category
        )
        pd.DataFrame(others_items).to_csv(other_items_file_path, index=False)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def plot_gpu_summary_time(task, base_path, category, name, data, title, threshold=1.0, time_percentage_col='Time (%)', operation_col='Operation', figsize=(6, 6)):
    try:
        # Prepare data
        data = data.values[0]
        
        # Convert to DataFrame
        df = convert_string_to_dataframe(data)
        
        # Plot pie chart
        save_path_pie_chart = get_visualization_path(
                base_path, f"{task}_gpu_summary_time_pie_chart.png", operation=f"{name}/gpu_summary_time", category = category
            )
        others_items_pie_chart = plot_pie_chart(df = df,
                    save_path=save_path_pie_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Avg (ns)',
                    stat_base = 'ns',
                    legend_title = operation_col,
                    operation_col = operation_col,
                    colors = sns.color_palette("tab20", len(df)),
                    plt_title = title,
                    figsize=figsize)
        
        save_path_bar_chart = get_visualization_path(
                base_path, f"{task}_gpu_summary_time_bar_chart.png", operation=f"{name}/gpu_summary_time", category = category
            )
        other_items_bar_chart = plot_bar_chart(df = df,
                    save_path=save_path_bar_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Avg (ns)',
                    stat_base = 'ns',
                    operation_col = operation_col,
                    x_label = operation_col,
                    y_label = time_percentage_col,
                    bar_label = time_percentage_col,
                    bar_color = 'b',
                    figsize = figsize,
                    legends = False,
                    bar_width = 0.35,
                    xlabel_rotation = 0)
        
        # Save other items to a file
        others_items = {
            "others_items_pie_chart": others_items_pie_chart,
            "other_items_bar_chart": other_items_bar_chart,
        }

        other_items_file_path = get_visualization_path(
            base_path, f"{task}_gpu_summary_time_other_items.csv", operation=f"{name}/gpu_summary_time", category = category
        )
        pd.DataFrame(others_items).to_csv(other_items_file_path, index=False)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def plot_gpu_summary_size(task, base_path, category, name, data, title, threshold=1.0, time_percentage_col='Size_Percent', operation_col='Operation', figsize=(6, 6)):
    try:
        # Prepare data
        data = data.values[0]
        
        # Convert to DataFrame
        df = convert_string_to_dataframe(data)
        
        # Plot pie chart
        save_path_pie_chart = get_visualization_path(
                base_path, f"{task}_gpu_summary_size_pie_chart.png", operation=f"{name}/gpu_summary_size", category = category
            )
        others_items_pie_chart = plot_pie_chart(df = df,
                    save_path=save_path_pie_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Avg (MB)',
                    stat_base = 'MB',
                    legend_title = operation_col,
                    operation_col = operation_col,
                    colors = sns.color_palette("tab20", len(df)),
                    plt_title = title,
                    figsize=figsize)
        
        # Plot bar chart
        save_path_bar_chart = get_visualization_path(
                base_path, f"{task}_gpu_summary_size_bar_chart.png", operation=f"{name}/gpu_summary_size", category = category
            )
        other_items_bar_chart = plot_bar_chart(df = df,
                    save_path=save_path_bar_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Avg (MB)',
                    stat_base = 'MB',
                    operation_col = operation_col,
                    x_label = operation_col,
                    y_label = time_percentage_col,
                    bar_label = time_percentage_col,
                    bar_color = 'b',
                    figsize = figsize,
                    legends = False,
                    bar_width = 0.35,
                    xlabel_rotation = 0)
        
        # Save other items to a file
        others_items = {
            "others_items_pie_chart": others_items_pie_chart,
            "other_items_bar_chart": other_items_bar_chart,
        }
        
        other_items_file_path = get_visualization_path(
            base_path, f"{task}_gpu_summary_size_other_items.csv", operation=f"{name}/gpu_summary_size", category = category
        )
        pd.DataFrame(others_items).to_csv(other_items_file_path, index=False)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def plot_kernel_functions_summary(task, base_path, category, name, data, title, threshold=1.0, time_percentage_col='Time (%)', operation_col='Operation', figsize=(6, 6)):
    try:
        # Prepare data
        data = data.values[0]
        
        # Convert to DataFrame
        df = convert_string_to_dataframe(data)
        
        # Plot pie chart
        save_path_pie_chart = get_visualization_path(
                base_path, f"{task}_kernel_functions_summary_pie_chart.png", operation=f"{name}/kernel_functions_summary", category = category
            )
        others_items_pie_chart = plot_pie_chart(df = df,
                    save_path=save_path_pie_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Time (s)',
                    stat_base = 's',
                    legend_title = operation_col,
                    operation_col = operation_col,
                    colors = sns.color_palette("tab20", len(df)),
                    plt_title = title,
                    figsize=(15, 15))
        
        # Plot bar chart
        save_path_bar_chart = get_visualization_path(
                base_path, f"{task}_kernel_functions_summary_bar_chart.png", operation=f"{name}/kernel_functions_summary", category = category
            )
        other_items_bar_chart = plot_bar_chart(df = df,
                    save_path=save_path_bar_chart,
                    title = title,
                    threshold = threshold,
                    percentage_col = time_percentage_col,
                    stat_col = 'Time (s)',
                    stat_base = 's',
                    operation_col = operation_col,
                    x_label = operation_col,
                    y_label = time_percentage_col,
                    bar_label = time_percentage_col,
                    bar_color = 'b',
                    figsize=(15, 15),
                    legends = False,
                    bar_width = 0.35,
                    xlabel_rotation = 0)
        
        # Save other items to a file
        others_items = {
            "others_items_pie_chart": others_items_pie_chart,
            "other_items_bar_chart": other_items_bar_chart,
        }
        
        other_items_file_path = get_visualization_path(
            base_path, f"{task}_kernel_functions_summary_other_items.csv", operation=f"{name}/kernel_functions_summary", category = category
        )
        pd.DataFrame(others_items).to_csv(other_items_file_path, index=False)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def analyze_api_call_frequency(task, base_path, category, name, df_block):
    try:
        cuda_api_summary = df_block['CUDA API Summary']
        
        if cuda_api_summary.empty:
            logger.info("No data found in CUDA API Summary")
            return
        
        df_cuda_api_summary = convert_string_to_dataframe(cuda_api_summary.values[0])
        
        if df_cuda_api_summary.empty:
            logger.info("No data found in CUDA API Summary")
            return
        
        api_frequency = df_cuda_api_summary[['Operation', 'Instances']]
        api_frequency = api_frequency.sort_values(by='Instances', ascending=False)
        
        logger.info("API Call Frequency Analysis:")
        logger.info(api_frequency.sort_values(by='Instances', ascending=False))

        # Save the data to a file
        api_frequency_file_path = get_visualization_path(
            base_path, f"{task}_block_api_frequency.csv", operation=f"{name}/api_frequency", category = category
        )
        api_frequency.to_csv(api_frequency_file_path, index=False)
        logger.info(f"Saved API Frequency Data to: {api_frequency_file_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def analyze_kernel_latency(task, base_path, category, name, df_block):
    try:
        kernels = df_block['Kernels']
        
        if kernels.empty:
            logger.info("No data found for Kernel Latency Analysis")
            return
        
        df_kernels = convert_string_to_dataframe(kernels.values[0])
        
        if df_kernels.empty:
            logger.info("No data found in Kernels")
            return
        
        latency_data = df_kernels[['Kernel Name', 'API Start (ns)', 'Kernel Start (ns)', 'Kernel Dur (ns)']]
        latency_data['Latency (ns)'] = latency_data['Kernel Start (ns)'] - latency_data['API Start (ns)']
        
        logger.info("Kernel Latency Analysis:")
        logger.info(latency_data.sort_values(by='Latency (ns)', ascending=False))
        
        # Save the data to a file
        kernel_latency_file_path = get_visualization_path(
            base_path, f"{task}_block_kernel_latency.csv", operation=f"{name}/kernel_latency", category = category
        )
        latency_data.to_csv(kernel_latency_file_path, index=False)
        logger.info(f"Saved Kernel Latency Data to: {kernel_latency_file_path}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def analyze_kernel_execution_distribution(task, base_path, category, name, df_block):
    try:
        kernel_summary = df_block['Kernel Summary (Short Names)']
        
        if kernel_summary.empty:
            logger.info("No data found for Kernel Execution Distribution Analysis")
            return
        
        df_kernel_summary = convert_string_to_dataframe(kernel_summary.values[0])
        
        if df_kernel_summary.empty:
            logger.info("No data found in Kernel Summary")
            return
        
        kernel_distribution = df_kernel_summary[['Kernel Name', 'Count']]
        kernel_distribution = kernel_distribution.sort_values(by='Count', ascending=False)
        
        logger.info("Kernel Execution Distribution Analysis:")
        logger.info(kernel_distribution.sort_values(by='Count', ascending=False))
        
        # Save the data to a file
        kernel_distribution_file_path = get_visualization_path(
            base_path, f"{task}_block_kernel_distribution.csv", operation=f"{name}/kernel_distribution", category = category
        )
        kernel_distribution.to_csv(kernel_distribution_file_path, index=False)
        logger.info(f"Saved Kernel Distribution Data to: {kernel_distribution_file_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def analyze_memory_bandwidth_utilization(task, base_path, category, name, df_block):
    try:
        memory_ops = df_block['Memory Ops']
        
        df_memory_ops = convert_string_to_dataframe(memory_ops.values[0])
        
        if df_memory_ops.empty:
            logger.info("No data found in Memory Ops")
            return
        
        bandwidth_data = df_memory_ops[['Name', 'Duration (ns)', 'Bytes (MB)']]
        bandwidth_data['Bandwidth (GB/s)'] = (bandwidth_data['Bytes (MB)'] * 1024) / (bandwidth_data['Duration (ns)'] * 1e-9)
        
        logger.info("Memory Bandwidth Utilization Analysis:")
        logger.info(bandwidth_data.sort_values(by='Bandwidth (GB/s)', ascending=False))

        # Save the data to a file
        bandwidth_file_path = get_visualization_path(
            base_path, f"{task}_block_bandwidth_utilization.csv", operation=f"{name}/bandwidth_utilization", category = category
        )
        bandwidth_data.to_csv(bandwidth_file_path, index=False)
        logger.info(f"Saved Bandwidth Utilization Data to: {bandwidth_file_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    

# Define a function to save filtered and plotted data for a specific level
def plot_and_save_by_level(task, base_path, category, name, df, level, level_name='Lvl'):
    try:
        df_level = df[df[level_name] == level]
        
        # If "Lvl" = 4 and "Name" = "0: TransformerBlock"
        # Keep only the following operations:
        # attention: Attention
        # attention_norm: RMSNorm
        # feed forward: FeedForward
        # ffn_norm: RMSNorm
        
        # Filter where Name does not contain "aten"
        if level == 4 and name == "0: TransformerBlock":
            df_level = df_level[~df_level['Name'].str.contains("aten")]
        
        if not df_level.empty:
            # Remove Time (%) column if it exists
            if 'Time (%)' in df_level.columns:
                df_level = df_level.drop(columns=['Time (%)'])
                
            # If Name contains ", op_id" then keep only the previous part
            # Previous Name : aten::mul, op_id = 3036	
            # Store OP_ID in a separate column
            df_level['OP_ID'] = df_level['Name'].str.extract(r'op_id = (\d+)')
            
            # New Name : aten::mul
            df_level['Name'] = df_level['Name'].str.split(', op_id').str[0]
            
            # Save the data for this level
            level_file_path = get_visualization_path(
                base_path, f"{task}_block_gpu_operations_level_{level}_rows.csv", operation=f"{name}/gpu_operations/level_{level}", category=category
            )
            df_level.to_csv(level_file_path, index=False)
            
            # Aggregate the data by Operation
            df_level_grouped = df_level.groupby('Name').agg({
                'Projected Duration (ns)': 'sum'
            }).reset_index()
            
            # Add a new column "Time (%)" to show the percentage of time taken by each operation
            df_level_grouped["Time (%)"] = (df_level_grouped["Projected Duration (ns)"] / df_level_grouped["Projected Duration (ns)"].sum()) * 100
            
            # Save the aggregated data to a file
            level_aggregated_file_path = get_visualization_path(
                base_path, f"{task}_block_gpu_operations_level_{level}_aggregated.csv", operation=f"{name}/gpu_operations/level_{level}", category=category
            )
            df_level_grouped.to_csv(level_aggregated_file_path, index=False)
            
            # Plot Pie Chart for the top GPU operations by time taken
            save_path_pie_chart = get_visualization_path(
                base_path, f"{task}_block_gpu_operations_level_{level}_pie_chart.png", operation=f"{name}/gpu_operations/level_{level}", category=category
            )
            other_items_pie_chart = plot_pie_chart(df=df_level_grouped,
                        save_path=save_path_pie_chart,
                        title=f"Top GPU Operations by Time Taken inside {name}",
                        threshold=1,
                        percentage_col='Time (%)',
                        stat_col='Projected Duration (ns)',
                        stat_base='ns',
                        legend_title="Operations",
                        operation_col="Name",
                        colors=sns.color_palette("tab20", len(df_level_grouped)),
                        plt_title=f"Top GPU Operations by Time Taken inside {name}",
                        figsize=(10, 10))
            
            # Save other items to a file
            other_items_file_path = get_visualization_path(
                base_path, f"{task}_block_gpu_operations_level_{level}_other_items.csv", operation=f"{name}/gpu_operations/level_{level}", category=category
            )
            pd.DataFrame(other_items_pie_chart).to_csv(other_items_file_path, index=False)

            logger.info(f"Filtered and Plotted GPU Operations for Level: {level}")
            
        else:
            logger.info(f"No data found for Level: {level}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def filter_and_plot_gpu_operations(task, base_path, category, name, df_nvtx_gpu_proj_trace_processed, level_name='Lvl'):
    try:
        # Filter the row from name to name
        # Find the indices of all occurrences of the given name
        name_indices = df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name].index
        
        if df_nvtx_gpu_proj_trace_processed.empty or len(name_indices) < 2:
            logger.error(f"No data found for Name: {name} or less than 2 occurrences found")
            return
        
        logger.info(f"Filtering and Plotting GPU Operations for: {name}")
        
        # Find the second occurrence of the given name
        start_index = name_indices[1]
        
        # Get the level of the given name
        lvl = df_nvtx_gpu_proj_trace_processed.loc[start_index, level_name]
        
        # Find the next occurrence of the same level after the start_index
        # Note: We exclude the start_index itself, hence the (start_index + 1)
        end_index = df_nvtx_gpu_proj_trace_processed[(df_nvtx_gpu_proj_trace_processed.index > start_index) & (df_nvtx_gpu_proj_trace_processed[level_name] == lvl)].index[0]
        
        logger.info(f"Start Index: {start_index}")
        logger.info(f"End Index: {end_index}")
        
        # Selecting the rows between the indices
        df_selected = df_nvtx_gpu_proj_trace_processed[(df_nvtx_gpu_proj_trace_processed.index > start_index) & (df_nvtx_gpu_proj_trace_processed.index < end_index)]
        
        # Save the data to a file
        selected_rows_file_path = get_visualization_path(
            base_path, f"{task}_block_gpu_operations_selected_rows.csv", operation=f"{name}/gpu_operations", category = category
        )
        df_selected.to_csv(selected_rows_file_path, index=False)
        
        # Add a new column "Time (%)" to show the percentage of time taken by each operation
        df_selected["Time (%)"] = (df_selected["Projected Duration (ns)"] / df_selected["Projected Duration (ns)"].sum()) * 100
        
        # Plot Pie Chart for the top 10 GPU operations by time taken "Proj_Avg_ns"
        save_path_pie_chart = get_visualization_path(
            base_path, f"{task}_block_gpu_operations_pie_chart.png", operation=f"{name}/gpu_operations", category = category
        )
        other_items_pie_chart = plot_pie_chart(df = df_selected,
                    save_path=save_path_pie_chart,
                    title = f"Top GPU Operations by Time Taken inside {name}",
                    threshold = 1,
                    percentage_col = 'Time (%)',
                    stat_col = 'Projected Duration (ns)',
                    stat_base = 'ns',
                    legend_title = "Operations",
                    operation_col = "Name",
                    colors = sns.color_palette("tab20", len(df_selected)),
                    plt_title = f"Top GPU Operations by Time Taken inside {name}",
                    figsize = (10, 10))
        
        # Save other items to a file
        other_items_file_path = get_visualization_path(
            base_path, f"{task}_block_gpu_operations_other_items.csv", operation=f"{name}/gpu_operations", category = category
        )
        pd.DataFrame(other_items_pie_chart).to_csv(other_items_file_path, index=False)
        
        logger.info("Filtered and Plotted GPU Operations")
        
        # Get the unique levels present in df_selected
        unique_levels = df_selected[level_name].unique()
        
        # Plot and save data for each level
        for level in unique_levels:
            plot_and_save_by_level(task, base_path, category, name, df_selected, level, level_name)
            
        logger.info("Filtered and Plotted GPU Operations by Level")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def visualize_operation_block(task, base_path, category, name, df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=False):
    try:
        logger.info("*"*150)
        # df_block = df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name].head(1)
        # If more than one block is found, select the second block, else select the first block
        if len(df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name]) > 1:
            logger.info(f"More than one block found for Name: {name}, Selecting the second block")
            df_block = df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name].iloc[1:2]
        else:
            logger.info(f"Only one block found for Name: {name}")
            df_block = df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name].head(1)
        logger.info(f"Visualizing Operation Block for: {name}")
        logger.info(f"DF Block: {df_block}")

        if df_block.empty:
            logger.error(f"No data found for Name: {name}")
            return
        
        # Filter and Plot GPU Operations
        if plot_gpu_operations:
            logger.info("Filtering and Plotting GPU Operations")
            filter_and_plot_gpu_operations(task, base_path, category, name, df_nvtx_gpu_proj_trace_processed)

        block_cuda_api_summary = df_block['CUDA API Summary']
        block_gpu_summary_time = df_block['Memory Summary by Time']
        block_gpu_summary_size = df_block['Memory Summary by Size']
        block_kernel_functions_summary = df_block['Kernel Summary (Short Names)']
        block_cuda_api_to_kernel_mapping = df_block['CUDA API to Kernel Mapping']
        block_cuda_api_to_memory_mapping = df_block['CUDA API to Memory Mapping']

        # CUDA API Summary
        logger.info("########################################################################################################")
        data = block_cuda_api_summary.values[0]
        df_block_cuda_api_summary = convert_string_to_dataframe(data)
        # Time (%)	Total Time (ns)	Instances	Avg (ns)	Med (ns)	Min (ns)	Max (ns)	StdDev (ns)	Operation
        if df_block_cuda_api_summary.empty:
            logger.info(f"No data found for {name} - CUDA API Summary")
            logger.info("No data found")
        else:
            # Print only Time (%), Total Time (ns), Instances, Avg (ns), Operation
            df_block_cuda_api_summary_sample = df_block_cuda_api_summary[['Time (%)', 'Total Time (ns)', 'Instances', 'Avg (ns)', 'Operation']]
            logger.info(f"\n{name} - CUDA API Summary")
            logger.info(df_block_cuda_api_summary_sample)
            
            # Plot CUDA API Summary
            plot_cuda_api_summary(task, base_path, category, name, block_cuda_api_summary, f"{name} - CUDA API Summary", figsize=(6, 6))
            # Save the data to a file
            cuda_api_summary_file_path = get_visualization_path(
                base_path, f"{task}_block_cuda_api_summary.csv", operation=f"{name}/cuda_api", category = category
            )
            df_block_cuda_api_summary.to_csv(cuda_api_summary_file_path, index=False)
            logger.info(f"Saved CUDA API Summary Data to: {cuda_api_summary_file_path}")
        
        logger.info("########################################################################################################")
        
        # GPU Summary Time
        data = block_gpu_summary_time.values[0]
        df_block_gpu_summary_time = convert_string_to_dataframe(data)
        if df_block_gpu_summary_time.empty:
            logger.info(f"\n{name} - GPU Summary Time")
            logger.info("No data found")
        else:
            logger.info(f"\n{name} - GPU Summary Time")
            logger.info(df_block_gpu_summary_time)
            
            # Plot GPU Summary Time
            plot_gpu_summary_time(task, base_path, category, name, block_gpu_summary_time, f"{name} - GPU Summary Time", figsize=(4, 4))
            # Save the data to a file
            gpu_summary_time_file_path = get_visualization_path(
                base_path, f"{task}_block_gpu_summary_time.csv", operation=f"{name}/gpu_summary_time", category = category
            )
            df_block_gpu_summary_time.to_csv(gpu_summary_time_file_path, index=False)
            logger.info(f"Saved GPU Summary Time Data to: {gpu_summary_time_file_path}")
            
        # GPU Summary Size
        data = block_gpu_summary_size.values[0]
        df_block_gpu_summary_size = convert_string_to_dataframe(data)
        
        if df_block_gpu_summary_size.empty:
            logger.info(f"\n{name} - GPU Summary Size")
            logger.info("No data found")
        else:
            logger.info(f"\n{name} - GPU Summary Size")
            logger.info(df_block_gpu_summary_size)
            
            # Plot GPU Summary Size
            plot_gpu_summary_size(task, base_path, category, name, block_gpu_summary_size, f"{name} - GPU Summary Size", figsize=(4, 4))
            # Save the data to a file
            gpu_summary_size_file_path = get_visualization_path(
                base_path, f"{task}_block_gpu_summary_size.csv", operation=f"{name}/gpu_summary_size", category = category
            )
            df_block_gpu_summary_size.to_csv(gpu_summary_size_file_path, index=False)
            logger.info(f"Saved GPU Summary Size Data to: {gpu_summary_size_file_path}")
            
        logger.info("########################################################################################################")
        
        # Kernel Functions Summary
        data = block_kernel_functions_summary.values[0]
        df_block_kernel_functions_summary = convert_string_to_dataframe(data)
        
        if df_block_kernel_functions_summary.empty:
            logger.info(f"\n{name} - Kernel Functions Summary")
            logger.info("No data found")
        else:
            sample_data = df_block_kernel_functions_summary[['Kernel Name', 'API Function', 'Count', 'KAvg (ns)']]
            logger.info(f"\n{name} - Kernel Functions Summary")
            logger.info(sample_data)
            
            # Save the data to a file
            kernel_functions_summary_file_path = get_visualization_path(
                base_path, f"{task}_block_kernel_functions_summary.csv", operation=f"{name}/kernel_functions_summary", category = category
            )
            df_block_kernel_functions_summary.to_csv(kernel_functions_summary_file_path, index=False)
            logger.info(f"Saved Kernel Functions Summary Data to: {kernel_functions_summary_file_path}")
            
        logger.info("########################################################################################################")
        
        # CUDA API to Kernel Mapping
        data = block_cuda_api_to_kernel_mapping.values[0]
        df_block_cuda_api_to_kernel_mapping = convert_string_to_dataframe(data)
        
        if df_block_cuda_api_to_kernel_mapping.empty:
            logger.info(f"\n{name} - CUDA API to Kernel Mapping")
            logger.info("No data found")
        else:
            df_block_cuda_api_to_kernel_mapping['Kernels'] = df_block_cuda_api_to_kernel_mapping['Kernels'].apply(lambda x: ', '.join(x))
            df_block_cuda_api_to_kernel_mapping = df_block_cuda_api_to_kernel_mapping.drop_duplicates()
            df_block_cuda_api_to_kernel_mapping['Kernels'] = df_block_cuda_api_to_kernel_mapping['Kernels'].apply(lambda x: map_kernel_name(x, unique_kernel_names))
            logger.info(f"\n{name} - CUDA API to Kernel Mapping")
            logger.info(df_block_cuda_api_to_kernel_mapping)
            
            # Save the data to a file
            cuda_api_to_kernel_mapping_file_path = get_visualization_path(
                base_path, f"{task}_block_cuda_api_to_kernel_mapping.csv", operation=f"{name}/cuda_api_to_kernel_mapping", category = category
            )
            df_block_cuda_api_to_kernel_mapping.to_csv(cuda_api_to_kernel_mapping_file_path, index=False)
            logger.info(f"Saved CUDA API to Kernel Mapping Data to: {cuda_api_to_kernel_mapping_file_path}")
        
        # CUDA API to Memory Mapping
        data = block_cuda_api_to_memory_mapping.values[0]
        df_block_cuda_api_to_memory_mapping = convert_string_to_dataframe(data)
        
        if df_block_cuda_api_to_memory_mapping.empty:
            logger.info(f"\n{name} - CUDA API to Memory Mapping")
            logger.info("No data found")
        else:
            logger.info(f"\n{name} - CUDA API to Memory Mapping")
            # display(df_block_cuda_api_to_memory_mapping)
            # Split multiple values in Memory Operations into separate rows and then drop duplicates
            df_block_cuda_api_to_memory_mapping = df_block_cuda_api_to_memory_mapping.explode('Memory Operations').drop_duplicates()
            logger.info(df_block_cuda_api_to_memory_mapping)
            
            # Save the data to a file
            cuda_api_to_memory_mapping_file_path = get_visualization_path(
                base_path, f"{task}_block_cuda_api_to_memory_mapping.csv", operation=f"{name}/cuda_api_to_memory_mapping", category = category
            )
            df_block_cuda_api_to_memory_mapping.to_csv(cuda_api_to_memory_mapping_file_path, index=False)
            logger.info(f"Saved CUDA API to Memory Mapping Data to: {cuda_api_to_memory_mapping_file_path}")
        
        logger.info("########################################################################################################")
            
        ######### NEW FUNCTIONS #########
        # API Call Frequency Analysis
        analyze_api_call_frequency(task, base_path, category, name, df_block)
        
        # Kernel Latency Analysis
        analyze_kernel_latency(task, base_path, category, name, df_block)
        
        # Kernel Execution Distribution Analysis
        analyze_kernel_execution_distribution(task, base_path, category, name, df_block)
        
        # Memory Bandwidth Utilization Analysis
        analyze_memory_bandwidth_utilization(task, base_path, category, name, df_block)
        
        logger.info("*"*150)
        
        return df_block

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def visualize_all_operations(task, category):
    try:
        nvtx_gpu_proj_trace_processed_path = get_reports_path(
            INFERENCE_NVTX_REPORTS_PATH, report_name="nvtx_gpu_proj_trace_processed.csv", category=category
        )
        df_nvtx_gpu_proj_trace_processed = pd.read_csv(nvtx_gpu_proj_trace_processed_path)
        logger.info(f"Loaded NVTX GPU Projection Trace Processed: {df_nvtx_gpu_proj_trace_processed.shape}, From: {nvtx_gpu_proj_trace_processed_path}")
        
        unique_kernel_names = get_unique_kernel_names_inference()
        
        operation_mapping = get_operation_mapping(INFERENCE_OPERATIONS_MAPPING)
        logger.info(f"Operation Mapping: {operation_mapping}")
                
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["initialize_model"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: initialize_model")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["load_weights"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: load_weights")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["load_checkpoint"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: load_checkpoint")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["model_load_state_dict"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: model_load_state_dict")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["model_quantization"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: model_quantization")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["quantize_based_on_type"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: quantize_based_on_type")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["move_model_to_cuda"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: move_model_to_cuda")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["text_completion"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: text_completion")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["transformer"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: transformer")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["token_embedding"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: token_embedding")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["transformer_block"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: transformer_block")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["attention"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: attention")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["attention_norm"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: attention_norm")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["wq"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: wq")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["wk"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: wk")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["wv"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: wv")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["wo"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: wo")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["feed_forward"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: feed_forward")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["w1"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: w1")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["w2"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: w2")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["w3"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: w3")
        
        # ffn_norm
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["ffn_norm"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names, plot_gpu_operations=True)
        logger.info("Visualized Operation Block for: ffn_norm")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e