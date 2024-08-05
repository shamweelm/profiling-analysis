import os
import pandas as pd
import seaborn as sns
import numpy as np
from profiling_analysis.visualize import plot_pie_chart, plot_bar_chart, plot_time
from profiling_analysis.helpers import get_visualization_path, get_summary_path
from profiling_analysis.configs.constants import (
    INFERENCE_CUDA_REPORTS_PATH,
    INFERENCE_NVTX_REPORTS_PATH,
)
from profiling_analysis import logger


def visualize_cuda_api_summaries(task, base_path, category=None):
    try:
        logger.info(f"Visualizing CUDA API Summaries for Task: {task}, Category: {category}, Base Path: {base_path}")
        cuda_api_summary_csv_path = get_summary_path(
            base_path, f"{task}_cuda_api_summary.csv", category
        )
        df_cupa_api_summary = pd.read_csv(cuda_api_summary_csv_path)
        logger.info(f"Loaded CUDA API Summary: {df_cupa_api_summary.shape}")

        save_path_pie_chart = get_visualization_path(
            base_path, f"{task}_cuda_api_summary_pie_chart.png", operation="cuda_api", category = category
        )
        others_items_pie_chart = plot_pie_chart(
            df=df_cupa_api_summary,
            save_path=save_path_pie_chart,
            title="CUDA API Summary Based on Percentage of Time consumed by each API",
            threshold=0.5,
            percentage_col="Time (%)",
            stat_col="Avg (ns)",
            stat_base="ns",
            legend_title="CUDA API Operations",
            operation_col="Operation",
            # colors = sns.color_palette("hsv", len(df_cupa_api_summary)),
            colors=sns.color_palette("tab20", len(df_cupa_api_summary)),
            plt_title="CUDA API Summary Based on Percentage of Time consumed by each API",
            figsize=(10, 10),
        )
        logger.info(f"Visualized CUDA API Summary Pie Chart and saved to: {save_path_pie_chart}")

        save_path_bar_chart = get_visualization_path(
            base_path, f"{task}_cuda_api_summary_bar_chart.png", operation="cuda_api", category = category
        )
        other_items_bar_chart = plot_bar_chart(
            df=df_cupa_api_summary,
            save_path=save_path_bar_chart,
            title="CUDA API Summary Based on Percentage of Time consumed by each API",
            threshold=0.5,
            percentage_col="Time (%)",
            stat_col="Avg (ns)",
            stat_base="ns",
            operation_col="Operation",
            x_label="CUDA API Operations",
            y_label="Time (%)",
            bar_label="Time (%)",
            bar_color="b",
            figsize=(8, 4),
            legends=False,
            bar_width=0.35,
            xlabel_rotation=0,
        )
        logger.info(f"Visualized CUDA API Summary Bar Chart and saved to: {save_path_bar_chart}")

        # Plot the top 5 CUDA API operations by time taken ('Avg (ns)')
        save_path_time_plot = get_visualization_path(
            base_path, f"{task}_cuda_api_summary_time_plot.png", operation="cuda_api", category = category
        )
        plot_time(
            df=df_cupa_api_summary,
            save_path=save_path_time_plot,
            title="Top 5 CUDA API Operations by Time Taken",
            limit=5,
            time_col="Avg (ns)",
            name_col="Operation",
            label_col="Avg (ns)",
            x_label="CUDA API Operations",
            y_label="Time (ns)",
            legend_label="Avg Time (ns)",
            bar_color="b",
            figsize=(12, 10),
            legends=False,
            bar_width=0.35,
            xlabel_rotation=0,
        )
        logger.info(f"Visualized CUDA API Summary Time Plot and saved to: {save_path_time_plot}")

        # Save other items to a file
        others_items = {
            "others_items_pie_chart": others_items_pie_chart,
            "other_items_bar_chart": other_items_bar_chart,
        }

        other_items_file_path = get_visualization_path(
            base_path, f"{task}_cuda_api_summary_other_items.csv", operation="cuda_api", category = category
        )
        pd.DataFrame(others_items).to_csv(other_items_file_path, index=False)
        logger.info(f"Saved other items to: {other_items_file_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def get_malloc_and_free_stats(task, base_path, category=None):
    try:
        logger.info(f"Getting CUDA Malloc and Free Stats for Task: {task}, Category: {category}, Base Path: {base_path}")
        cuda_api_trace_filtered_csv_path = get_summary_path(
            base_path, f"{task}_cuda_api_trace_filtered.csv", category
        )
        df_cuda_api_trace_filtered = pd.read_csv(cuda_api_trace_filtered_csv_path)
        logger.info(f"Loaded CUDA API Trace Filtered: {df_cuda_api_trace_filtered.shape}")

        # Get all instances of "cudaMalloc" operations
        df_cuda_malloc = df_cuda_api_trace_filtered[
            df_cuda_api_trace_filtered["Name"] == "cudaMalloc"
        ]

        # Total duration of cudaMalloc calls
        total_duration = df_cuda_malloc["Duration (ns)"].sum()

        # Average duration of cudaMalloc calls
        average_duration = df_cuda_malloc["Duration (ns)"].mean()

        # Total number of cudaMalloc calls
        total_calls = df_cuda_malloc.shape[0]

        # print(f"Total Duration of cudaMalloc calls: {total_duration} ns")
        # print(f"Average Duration of cudaMalloc calls: {average_duration} ns")
        # print(f"Total Number of cudaMalloc calls: {total_calls}")
        # Convert to DataFrame and save to a file
        cuda_malloc_stats = {
            "Total Duration": total_duration,
            "Average Duration": average_duration,
            "Total Calls": total_calls,
        }
        df_cuda_malloc_stats = pd.DataFrame([cuda_malloc_stats])
        cuda_malloc_stats_file_path = get_summary_path(
            base_path, f"{task}_cuda_malloc_stats.csv", category
        )
        df_cuda_malloc_stats.to_csv(cuda_malloc_stats_file_path, index=False)
        logger.info(f"Saved CUDA Malloc Stats to: {cuda_malloc_stats_file_path}")
        
        # Get all instances of "cudaFree" operations
        df_cuda_free = df_cuda_api_trace_filtered[
            df_cuda_api_trace_filtered["Name"] == "cudaFree"
        ]

        # Total duration of cudaFree calls
        total_duration = df_cuda_free["Duration (ns)"].sum()

        # Average duration of cudaFree calls
        average_duration = df_cuda_free["Duration (ns)"].mean()

        # Total number of cudaFree calls
        total_calls = df_cuda_free.shape[0]

        # print(f"Total Duration of cudaFree calls: {total_duration} ns")
        # print(f"Average Duration of cudaFree calls: {average_duration} ns")
        # print(f"Total Number of cudaFree calls: {total_calls}")

        # Convert to DataFrame and save to a file
        cuda_free_stats = {
            "Total Duration": total_duration,
            "Average Duration": average_duration,
            "Total Calls": total_calls,
        }

        df_cuda_free_stats = pd.DataFrame([cuda_free_stats])
        cuda_free_stats_file_path = get_summary_path(
            base_path, f"{task}_cuda_free_stats.csv", category
        )
        df_cuda_free_stats.to_csv(cuda_free_stats_file_path, index=False)
        logger.info(f"Saved CUDA Free Stats to: {cuda_free_stats_file_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def visualize_cuda_memtime_summary(task, base_path, category=None):
    try:
        logger.info(f"Visualizing CUDA Memory Time Summary for Task: {task}, Category: {category}, Base Path: {base_path}")
        cuda_memtime_summary_csv_path = get_summary_path(
            base_path, f"{task}_cuda_memtime_summary.csv", category
        )
        df_cuda_memtime_summary = pd.read_csv(cuda_memtime_summary_csv_path)
        logger.info(f"Loaded CUDA Memory Time Summary: {df_cuda_memtime_summary.shape}")

        save_path_pie_chart = get_visualization_path(
            base_path, f"{task}_cuda_memtime_summary_pie_chart.png", operation="cuda_memtime", category = category
        )
        other_items = plot_pie_chart(
            df=df_cuda_memtime_summary,
            save_path=save_path_pie_chart,
            title="CUDA Memory Time Summary",
            threshold=0.000,
            percentage_col="Time (%)",
            stat_col="Avg (ns)",
            stat_base="ns",
            legend_title="CUDA Memory Operations",
            operation_col="Operation",
            # Color with brown color palette other than tab20
            colors=sns.color_palette("Dark2", len(df_cuda_memtime_summary)),
            plt_title="CUDA Memory Time Summary",
            figsize=(10, 10),
        )
        logger.info(f"Visualized CUDA Memory Time Summary Pie Chart and saved to: {save_path_pie_chart}")

        # Save other items to a file
        other_items_file_path = get_visualization_path(
            base_path, f"{task}_cuda_memtime_summary_other_items.csv", operation="cuda_memtime", category = category
        )
        pd.DataFrame(other_items).to_csv(other_items_file_path, index=False)
        logger.info(f"Saved other items to: {other_items_file_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def visualize_cuda_memsize_summary(task, base_path, category=None):
    try:
        logger.info(f"Visualizing CUDA Memory Size Summary for Task: {task}, Category: {category}, Base Path: {base_path}")
        cuda_memsize_summary_csv_path = get_summary_path(
            base_path, f"{task}_cuda_memsize_summary.csv", category
        )
        df_cuda_memsize_summary = pd.read_csv(cuda_memsize_summary_csv_path)
        logger.info(f"Loaded CUDA Memory Size Summary: {df_cuda_memsize_summary.shape}")

        save_path_pie_chart = get_visualization_path(
            base_path, f"{task}_cuda_memsize_summary_pie_chart.png", operation="cuda_memsize", category = category
        )
        other_items = plot_pie_chart(
            df=df_cuda_memsize_summary,
            save_path=save_path_pie_chart,
            title="CUDA Memory Size Summary",
            threshold=0.0000,
            percentage_col="Size_Percent",
            stat_col="Avg (MB)",
            stat_base="MB",
            legend_title="CUDA Memory Operations",
            operation_col="Operation",
            # Color with brown color palette other than tab20
            colors=sns.color_palette("Dark2", len(df_cuda_memsize_summary)),
            plt_title="CUDA Memory Size Summary",
            figsize=(10, 10),
        )
        logger.info(f"Visualized CUDA Memory Size Summary Pie Chart and saved to: {save_path_pie_chart}")

        # Save other items to a file
        other_items_file_path = get_visualization_path(
            base_path, f"{task}_cuda_memsize_summary_other_items.json", operation="cuda_memsize", category = category
        )
        pd.DataFrame(other_items).to_json(other_items_file_path)
        logger.info(f"Saved other items to: {other_items_file_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def visualize_kernel_summaries(task, base_path, category=None):
    try:
        logger.info(f"Visualizing CUDA Kernel Summaries for Task: {task}, Category: {category}, Base Path: {base_path}")
        cuda_kernel_launch_exec_short_summary_csv_path = get_summary_path(
            base_path, f"{task}_cuda_kernel_launch_exec_short_summary.csv", category
        )
        df_cuda_kernel_launch_exec_short_summary = pd.read_csv(
            cuda_kernel_launch_exec_short_summary_csv_path
        )
        logger.info(f"Loaded CUDA Kernel Launch and Execution Short Summary: {df_cuda_kernel_launch_exec_short_summary.shape}")

        # Save the top 10 CUDA Kernel operations by time taken ('KAvg (ns)') in DataFrame
        df_top_10_cuda_kernel_launch_exec_short_summary = (
            df_cuda_kernel_launch_exec_short_summary.sort_values(
                by="KAvg (ns)", ascending=False
            ).head(10)
        )
        df_top_10_cuda_kernel_launch_exec_short_summary = (
            df_top_10_cuda_kernel_launch_exec_short_summary[
                [
                    "Kernel Name",
                    "Count",
                    "KAvg (ns)",
                    "AAvg (ns)",
                    "TAvg (ns)",
                    "QAvg (ns)",
                    "API Function",
                ]
            ]
        )
        df_top_10_cuda_kernel_launch_exec_short_summary.index = np.arange(
            1, len(df_top_10_cuda_kernel_launch_exec_short_summary) + 1
        )

        # Save the top 10 CUDA Kernel operations by time taken ('KAvg (ns)') in a file
        top_10_cuda_kernel_launch_exec_short_summary_file_path = get_summary_path(
            base_path,
            f"{task}_top_10_cuda_kernel_launch_exec_short_summary.csv",
            category,
        )
        # Add Time (%) column
        df_top_10_cuda_kernel_launch_exec_short_summary["Time (%)"] = (
            df_top_10_cuda_kernel_launch_exec_short_summary["KAvg (ns)"]
            / df_top_10_cuda_kernel_launch_exec_short_summary["KAvg (ns)"].sum()
        ) * 100
        df_top_10_cuda_kernel_launch_exec_short_summary.to_csv(
            top_10_cuda_kernel_launch_exec_short_summary_file_path, index=False
        )
        logger.info(f"Saved Top 10 CUDA Kernel Launch and Execution Short Summary to: {top_10_cuda_kernel_launch_exec_short_summary_file_path}")

        # Plot the top 10 CUDA Kernel operations by time taken ('KAvg (ns)')
        save_path_pie_chart = get_visualization_path(
            base_path,
            f"{task}_top_10_cuda_kernel_launch_exec_short_summary_pie_chart.png",
            operation="cuda_kernel",
            category=category,
        )
        other_items = plot_pie_chart(
            df=df_top_10_cuda_kernel_launch_exec_short_summary,
            save_path=save_path_pie_chart,
            title="CUDA Kernel Launch and Execution Time Summary",
            threshold=0.5,
            percentage_col="Time (%)",
            stat_col="KAvg (ns)",
            stat_base="ns",
            legend_title="CUDA Kernel Operations",
            operation_col="Kernel Name",
            colors=sns.color_palette(
                "tab20", len(df_top_10_cuda_kernel_launch_exec_short_summary)
            ),
            plt_title="CUDA Kernel Launch and Execution Time Summary",
            figsize=(10, 10),
        )
        logger.info(f"Visualized Top 10 CUDA Kernel Launch and Execution Short Summary Pie Chart and saved to: {save_path_pie_chart}")

        # Save other items to a file
        other_items_file_path = get_visualization_path(
            base_path,
            f"{task}_cuda_kernel_launch_exec_short_summary_other_items.csv",
            operation="cuda_kernel",
            category=category,
        )
        pd.DataFrame(other_items).to_csv(other_items_file_path, index=False)
        logger.info(f"Saved other items to: {other_items_file_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def visualize_all_summaries_for_inference(task, category=None):
    try:
        logger.info(f"Visualizing All Summaries for Inference Task: {task}, Category: {category}")
        visualize_cuda_api_summaries(task, INFERENCE_CUDA_REPORTS_PATH, category)
        logger.info("Visualized CUDA API Summaries")
        visualize_cuda_memtime_summary(task, INFERENCE_CUDA_REPORTS_PATH, category)
        logger.info("Visualized CUDA Memory Time Summary")
        visualize_cuda_memsize_summary(task, INFERENCE_CUDA_REPORTS_PATH, category)
        logger.info("Visualized CUDA Memory Size Summary")
        visualize_kernel_summaries(task, INFERENCE_CUDA_REPORTS_PATH, category)
        logger.info("Visualized CUDA Kernel Summaries")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
