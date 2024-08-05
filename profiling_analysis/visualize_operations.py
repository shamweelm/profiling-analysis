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
                base_path, f"{task}_cuda_api_summary_pie_chart.png", operation=f"{name}_cuda_api", category = category
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
                base_path, f"{task}_cuda_api_summary_bar_chart.png", operation=f"{name}_cuda_api", category = category
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
            base_path, f"{task}_cuda_api_summary_other_items.csv", operation=f"{name}_cuda_api", category = category
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
                base_path, f"{task}_gpu_summary_time_pie_chart.png", operation=f"{name}_gpu_summary_time", category = category
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
                base_path, f"{task}_gpu_summary_time_bar_chart.png", operation=f"{name}_gpu_summary_time", category = category
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
            base_path, f"{task}_gpu_summary_time_other_items.csv", operation=f"{name}_gpu_summary_time", category = category
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
                base_path, f"{task}_gpu_summary_size_pie_chart.png", operation=f"{name}_gpu_summary_size", category = category
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
                base_path, f"{task}_gpu_summary_size_bar_chart.png", operation=f"{name}_gpu_summary_size", category = category
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
            base_path, f"{task}_gpu_summary_size_other_items.csv", operation=f"{name}_gpu_summary_size", category = category
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
                base_path, f"{task}_kernel_functions_summary_pie_chart.png", operation=f"{name}_kernel_functions_summary", category = category
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
                base_path, f"{task}_kernel_functions_summary_bar_chart.png", operation=f"{name}_kernel_functions_summary", category = category
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
            base_path, f"{task}_kernel_functions_summary_other_items.csv", operation=f"{name}_kernel_functions_summary", category = category
        )
        pd.DataFrame(others_items).to_csv(other_items_file_path, index=False)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def analyze_api_call_frequency(df_block):
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
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def analyze_kernel_latency(df_block):
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
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def analyze_kernel_execution_distribution(df_block):
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

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def analyze_memory_bandwidth_utilization(df_block):
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
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
def visualize_operation_block(task, base_path, category, name, df_nvtx_gpu_proj_trace_processed, unique_kernel_names):
    try:
        # df_block = df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name].head(1)
        df_block = df_nvtx_gpu_proj_trace_processed[df_nvtx_gpu_proj_trace_processed['Name'] == name].iloc[1:2]
        logger.info(f"Visualizing Operation Block for: {name}")
        logger.info(f"DF Block: {df_block}")

        if df_block.empty:
            logger.error(f"No data found for Name: {name}")
            return

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
        
        logger.info("########################################################################################################")
            
        ######### NEW FUNCTIONS #########
        # API Call Frequency Analysis
        analyze_api_call_frequency(df_block)
        
        # Kernel Latency Analysis
        analyze_kernel_latency(df_block)
        
        # Kernel Execution Distribution Analysis
        analyze_kernel_execution_distribution(df_block)
        
        # Memory Bandwidth Utilization Analysis
        analyze_memory_bandwidth_utilization(df_block)
        
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
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["token_embedding"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: token_embedding")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["transformer_block"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: transformer_block")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["attention_norm"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: attention_norm")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["attention"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: attention")
        
        visualize_operation_block(task, INFERENCE_OPERATIONS_PATH, category, operation_mapping["feed_forward"], df_nvtx_gpu_proj_trace_processed, unique_kernel_names)
        logger.info("Visualized Operation Block for: feed_forward")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e