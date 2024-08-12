from datetime import datetime
import os
import numpy as np
from profiling_analysis.configs.constants import (
    INFERENCE_CUDA_REPORTS_PATH,
    INFERENCE_NVTX_REPORTS_PATH,
)
from profiling_analysis.helpers import get_summary_path
from profiling_analysis import logger


def summarize_cuda_api_operations(df):
    """Create a detailed summary for CUDA API operations."""
    try:
        logger.info("Summarizing CUDA API operations...")
        # Group by operation type and calculate required statistics till 5 decimal places
        operation_summary = (
            df.groupby("Name")
            .agg(
                Total_Time_ns=("Duration (ns)", lambda x: round(x.sum(), 6)),
                Instances=("Duration (ns)", "size"),
                Avg_ns=("Duration (ns)", lambda x: round(x.mean(), 6)),
                Med_ns=("Duration (ns)", lambda x: round(x.median(), 6)),
                Min_ns=("Duration (ns)", lambda x: round(x.min(), 6)),
                Max_ns=("Duration (ns)", lambda x: round(x.max(), 6)),
                StdDev_ns=("Duration (ns)", lambda x: round(x.std(), 6)),
            )
            .reset_index()
        )

        # Calculate the total time of all operations to find the percentage of time spent
        total_time = operation_summary["Total_Time_ns"].sum()
        operation_summary["Time_Percent"] = (
            operation_summary["Total_Time_ns"] / total_time
        ) * 100
        logger.info(f"Calculated Time Percentages for CUDA API operations")

        # Reorder and rename columns for the final summary
        operation_summary = operation_summary[
            [
                "Time_Percent",
                "Total_Time_ns",
                "Instances",
                "Avg_ns",
                "Med_ns",
                "Min_ns",
                "Max_ns",
                "StdDev_ns",
                "Name",
            ]
        ]
        operation_summary.columns = [
            "Time (%)",
            "Total Time (ns)",
            "Instances",
            "Avg (ns)",
            "Med (ns)",
            "Min (ns)",
            "Max (ns)",
            "StdDev (ns)",
            "Operation",
        ]

        # Categorize operations
        operation_summary["Category"] = np.where(
            operation_summary["Operation"].str.contains(
                "cudaMalloc|cudaFree|cudaMemcpy"
            ),
            "MEMORY_OPER",
            "CUDA_API",
        )
        logger.info(f"Added Columns for Categorization of CUDA API operations")

        operation_summary_sorted = operation_summary.sort_values(
            by="Time (%)", ascending=False
        )

        return operation_summary_sorted

    except Exception as e:
        logger.error(f"Error in summarize_cuda_api_operations: {e}")
        raise e


# Function to summarize memory operations by size
def summarize_memory_by_size(df, operation_col="Name"):
    try:
        logger.info("Summarizing memory operations by size...")
        # Filter only memory-related operations
        # df = df[df[operation_col].str.contains('memcpy|memset', case=False, na=False)]
        # Filter where Bytes (MB) is not null
        df = df[~df["Bytes (MB)"].isnull()]

        memory_summary = (
            df.groupby(operation_col)
            .agg(
                Total_MB=("Bytes (MB)", lambda x: round(x.sum(), 12)),
                Count=("Bytes (MB)", "size"),
                Avg_MB=("Bytes (MB)", lambda x: round(x.mean(), 12)),
                Med_MB=("Bytes (MB)", lambda x: round(x.median(), 12)),
                Min_MB=("Bytes (MB)", lambda x: round(x.min(), 12)),
                Max_MB=("Bytes (MB)", lambda x: round(x.max(), 12)),
                StdDev_MB=("Bytes (MB)", lambda x: round(x.std(), 12)),
            )
            .reset_index()
        )

        # Get percentage of total memory transferred
        total_memory = memory_summary["Total_MB"].sum()
        memory_summary["Size_Percent"] = (
            memory_summary["Total_MB"] / total_memory
        ) * 100

        logger.info(f"Calculated Size Percentages for memory operations")

        memory_summary.columns = [
            "Operation",
            "Total (MB)",
            "Count",
            "Avg (MB)",
            "Med (MB)",
            "Min (MB)",
            "Max (MB)",
            "StdDev (MB)",
            "Size_Percent",
        ]
        logger.info(f"Renamed Columns for memory operations")

        return memory_summary

    except Exception as e:
        logger.error(f"Error in summarize_memory_by_size: {e}")
        raise e


# Function to summarize memory operations by time
def summarize_memory_by_time(df, operation_col="Name"):
    try:
        logger.info("Summarizing memory operations by time...")
        # Filter only memory-related operations
        # df = df[df[operation_col].str.contains('memcpy|memset', case=False, na=False)]
        df = df[~df["Bytes (MB)"].isnull()]

        time_summary = (
            df.groupby(operation_col)
            .agg(
                Total_Time_ns=("Duration (ns)", lambda x: round(x.sum(), 6)),
                Count=("Duration (ns)", "size"),
                Avg_ns=("Duration (ns)", lambda x: round(x.mean(), 6)),
                Med_ns=("Duration (ns)", lambda x: round(x.median(), 6)),
                Min_ns=("Duration (ns)", lambda x: round(x.min(), 6)),
                Max_ns=("Duration (ns)", lambda x: round(x.max(), 6)),
                StdDev_ns=("Duration (ns)", lambda x: round(x.std(), 6)),
            )
            .reset_index()
        )

        # Calculate the total time to find the percentage of time spent
        total_time = time_summary["Total_Time_ns"].sum()
        time_summary["Time_Percent"] = (
            time_summary["Total_Time_ns"] / total_time
        ) * 100
        logger.info(f"Calculated Time Percentages for memory operations")

        time_summary.rename(
            columns={
                "Time_Percent": "Time (%)",
                "Total_Time_ns": "Total Time (ns)",
                "Count": "Count",
                "Avg_ns": "Avg (ns)",
                "Med_ns": "Med (ns)",
                "Min_ns": "Min (ns)",
                "Max_ns": "Max (ns)",
                "StdDev_ns": "StdDev (ns)",
                operation_col: "Operation",
            },
            inplace=True,
        )
        logger.info(f"Renamed Columns for memory operations")

        return time_summary

    except Exception as e:
        logger.error(f"Error in summarize_memory_by_time: {e}")
        raise e


def summarize_kernel_launch_and_exec(df, short_kernel_name=False):
    try:
        logger.info(
            f"Summarizing kernel launch and execution times with Short Kernel Name : {short_kernel_name}..."
        )
        
        if df.empty or df is None:
            logger.warning("Empty DataFrame for Kernel Execution Trace")
            return df
        
        # Columns
        # API Start (ns)	API Dur (ns)	Queue Start (ns)	Queue Dur (ns)	Kernel Start (ns)	Kernel Dur (ns)	Total Dur (ns)	PID	TID	DevId	API Function	GridXYZ	BlockXYZ	Kernel Name

        # Determine which kernel name column to use
        kernel_name_col = "Short Kernel Name" if short_kernel_name else "Kernel Name"

        # Group by PID, TID, DevId, API Function, and the appropriate Kernel Name column
        kernel_summary = (
            df.groupby(["PID", "TID", "DevId", "API Function", kernel_name_col])
            .agg(
                Count=("API Start (ns)", "size"),
                QCount=("Queue Dur (ns)", "size"),
                TAvg=("Total Dur (ns)", "mean"),
                TMed=("Total Dur (ns)", "median"),
                TMin=("Total Dur (ns)", "min"),
                TMax=("Total Dur (ns)", "max"),
                TStdDev=("Total Dur (ns)", "std"),
                AAvg=("API Dur (ns)", "mean"),
                AMed=("API Dur (ns)", "median"),
                AMin=("API Dur (ns)", "min"),
                AMax=("API Dur (ns)", "max"),
                AStdDev=("API Dur (ns)", "std"),
                QAvg=("Queue Dur (ns)", "mean"),
                QMed=("Queue Dur (ns)", "median"),
                QMin=("Queue Dur (ns)", "min"),
                QMax=("Queue Dur (ns)", "max"),
                QStdDev=("Queue Dur (ns)", "std"),
                KAvg=("Kernel Dur (ns)", "mean"),
                KMed=("Kernel Dur (ns)", "median"),
                KMin=("Kernel Dur (ns)", "min"),
                KMax=("Kernel Dur (ns)", "max"),
                KStdDev=("Kernel Dur (ns)", "std"),
            )
            .reset_index()
        )

        # Reorder and rename columns for the final summary
        kernel_summary = kernel_summary[
            [
                "PID",
                "TID",
                "DevId",
                "Count",
                "QCount",
                "TAvg",
                "TMed",
                "TMin",
                "TMax",
                "TStdDev",
                "AAvg",
                "AMed",
                "AMin",
                "AMax",
                "AStdDev",
                "QAvg",
                "QMed",
                "QMin",
                "QMax",
                "QStdDev",
                "KAvg",
                "KMed",
                "KMin",
                "KMax",
                "KStdDev",
                "API Function",
                kernel_name_col,
            ]
        ]

        kernel_summary.columns = [
            "PID",
            "TID",
            "DevId",
            "Count",
            "QCount",
            "TAvg (ns)",
            "TMed (ns)",
            "TMin (ns)",
            "TMax (ns)",
            "TStdDev (ns)",
            "AAvg (ns)",
            "AMed (ns)",
            "AMin (ns)",
            "AMax (ns)",
            "AStdDev (ns)",
            "QAvg (ns)",
            "QMed (ns)",
            "QMin (ns)",
            "QMax (ns)",
            "QStdDev (ns)",
            "KAvg (ns)",
            "KMed (ns)",
            "KMin (ns)",
            "KMax (ns)",
            "KStdDev (ns)",
            "API Function",
            "Kernel Name",
        ]

        # Sort by Total Average Duration
        kernel_summary = kernel_summary.sort_values(by="TAvg (ns)", ascending=False)

        logger.info(f"Sorted Kernel Summary by Total Average Duration")

        return kernel_summary

    except Exception as e:
        logger.error(f"Error in summarize_kernel_launch_and_exec: {e}")
        raise e


def save_summary_to_csv(df, base_path, task, operation, category=None):
    try:
        summary_csv_path = get_summary_path(
            base_path, f"{task}_{operation}_summary.csv", category
        )
        df.to_csv(summary_csv_path, index=False)

        logger.info(f"Saved Summary to {summary_csv_path}")

    except Exception as e:
        logger.error(f"Error in save_summary_to_csv: {e}")
        raise e


def generate_cuda_summaries(
    task,
    df_cuda_api_trace_filtered,
    df_cuda_gpu_trace_filtered,
    df_cuda_kernel_exec_trace,
    category=None,
):
    try:
        logger.info("Generating CUDA Summaries...")
        # Generate CUDA API summaries
        df_cuda_api_summary = summarize_cuda_api_operations(df_cuda_api_trace_filtered)
        save_summary_to_csv(
            df_cuda_api_summary, INFERENCE_CUDA_REPORTS_PATH, task, "cuda_api", category
        )
        logger.info("Generated CUDA API Summaries")

        # Generate memory summaries
        logger.info("Generating Memory Summaries...")
        df_cuda_memtime_summary = summarize_memory_by_time(df_cuda_gpu_trace_filtered)
        df_cuda_memsize_summary = summarize_memory_by_size(df_cuda_gpu_trace_filtered)
        # Save the summary to a CSV file
        save_summary_to_csv(
            df_cuda_memtime_summary,
            INFERENCE_CUDA_REPORTS_PATH,
            task,
            "cuda_memtime",
            category,
        )
        save_summary_to_csv(
            df_cuda_memsize_summary,
            INFERENCE_CUDA_REPORTS_PATH,
            task,
            "cuda_memsize",
            category,
        )
        logger.info("Generated Memory Summaries")

        # Generate kernel launch and execution summaries
        if (df_cuda_kernel_exec_trace is None) or (df_cuda_kernel_exec_trace.empty):
            logger.warning("No Kernel Execution Trace found")
            return
        else:
            logger.info("Generating Kernel Launch and Execution Summaries...")
            df_cuda_kernel_launch_and_exec_time = summarize_kernel_launch_and_exec(
                df_cuda_kernel_exec_trace
            )
            save_summary_to_csv(
                df_cuda_kernel_launch_and_exec_time,
                INFERENCE_CUDA_REPORTS_PATH,
                task,
                "cuda_kernel_launch_exec",
                category,
            )
            logger.info("Generated Kernel Launch and Execution Summaries")
            # Save to "generated_reports" folder

            # Generate a summary with short kernel names
            logger.info(
                "Generating Kernel Launch and Execution Summaries with Short Kernel Names..."
            )
            df_cuda_kernel_launch_and_exec_time_short = summarize_kernel_launch_and_exec(
                df_cuda_kernel_exec_trace, short_kernel_name=True
            )
            save_summary_to_csv(
                df_cuda_kernel_launch_and_exec_time_short,
                INFERENCE_CUDA_REPORTS_PATH,
                task,
                "cuda_kernel_launch_exec_short",
                category,
            )
            logger.info(
                "Generated Kernel Launch and Execution Summaries with Short Kernel Names"
            )

    except Exception as e:
        logger.error(f"Error in generate_cuda_summaries: {e}")
        raise e


def summarize_nvtx_gpu_projection(df_nvtx):
    try:
        logger.info("Summarizing NVTX GPU Projection operations...")
        # Assuming df_nvtx has columns such as: Range, Style, Total Proj Time (ns), etc.
        # Group by 'Range' and 'Style' and calculate statistics
        projection_summary = (
            df_nvtx.groupby(["Name", "Style"])
            .agg(
                Total_Proj_Time_ns=("Projected Duration (ns)", "sum"),
                Total_Range_Time_ns=("Orig Duration (ns)", "sum"),
                Range_Instances=("RangeId", "nunique"),
                Proj_Avg_ns=("Projected Duration (ns)", "mean"),
                Proj_Med_ns=("Projected Duration (ns)", "median"),
                Proj_Min_ns=("Projected Duration (ns)", "min"),
                Proj_Max_ns=("Projected Duration (ns)", "max"),
                Proj_StdDev_ns=("Projected Duration (ns)", "std"),
                Total_GPU_Ops=("NumGPUOps", "sum"),
                Avg_GPU_Ops=("NumGPUOps", "mean"),
                Avg_Range_Lvl=("Lvl", "mean"),
                Avg_Num_Child=("NumChild", "mean"),
            )
            .reset_index()
        )

        # Handling NaN values for standard deviation in cases of single data point
        projection_summary["Proj_StdDev_ns"].fillna(0, inplace=True)
        logger.info(f"Calculated Statistics for NVTX GPU Projection operations")

        # Additional formatting or calculations can be added here as needed

        return projection_summary

    except Exception as e:
        logger.error(f"Error in summarize_nvtx_gpu_projection: {e}")
        raise e


def summarize_nvtx_pushpop(df):
    try:
        logger.info("Summarizing NVTX PushPop operations...")
        # Calculate total duration and create a derived column for it
        df["Duration (ns)"] = df["End (ns)"] - df["Start (ns)"]

        # Group by 'Range' and calculate the necessary statistics
        summary = (
            df.groupby("Name")
            .agg(
                Total_Time_ns=("Duration (ns)", lambda x: round(x.sum(), 6)),
                Instances=("Duration (ns)", "size"),
                Avg_ns=("Duration (ns)", lambda x: round(x.mean(), 6)),
                Med_ns=("Duration (ns)", lambda x: round(x.median(), 6)),
                Min_ns=("Duration (ns)", lambda x: round(x.min(), 6)),
                Max_ns=("Duration (ns)", lambda x: round(x.max(), 6)),
                StdDev_ns=("Duration (ns)", lambda x: round(x.std(), 6)),
            )
            .reset_index()
        )

        # Calculate total time across all operations to find the percentage of time spent
        total_time = summary["Total_Time_ns"].sum()
        summary["Time_Percent"] = (summary["Total_Time_ns"] / total_time) * 100
        logger.info(f"Calculated Time Percentages for NVTX PushPop operations")

        # Handling NaN values for standard deviation in cases of single data point
        summary["StdDev_ns"].fillna(0, inplace=True)

        # Format the output to match your desired output
        summary = summary[
            [
                "Time_Percent",
                "Total_Time_ns",
                "Instances",
                "Avg_ns",
                "Med_ns",
                "Min_ns",
                "Max_ns",
                "StdDev_ns",
                "Name",
            ]
        ]
        summary.columns = [
            "Time (%)",
            "Total Time (ns)",
            "Instances",
            "Avg (ns)",
            "Med (ns)",
            "Min (ns)",
            "Max (ns)",
            "StdDev (ns)",
            "Range",
        ]
        logger.info(f"Formatted Summary for NVTX PushPop operations")

        return summary

    except Exception as e:
        logger.error(f"Error in summarize_nvtx_pushpop: {e}")
        raise e


def generate_nvtx_summaries(
    task, df_nvtx_gpu_proj_trace_filtered, df_nvtx_pushpop_trace_filtered, category=None
):
    try:
        logger.info("Generating NVTX Summaries...")
        # Generate NVTX GPU Projection summaries
        df_nvtx_gpu_proj_summary = summarize_nvtx_gpu_projection(
            df_nvtx_gpu_proj_trace_filtered
        )
        save_summary_to_csv(
            df_nvtx_gpu_proj_summary,
            INFERENCE_NVTX_REPORTS_PATH,
            task,
            "nvtx_gpu_proj",
            category,
        )
        logger.info("Generated NVTX GPU Projection Summaries")

        # Generate NVTX PushPop summaries
        logger.info("Generating NVTX PushPop Summaries...")
        df_nvtx_pushpop_summary = summarize_nvtx_pushpop(df_nvtx_pushpop_trace_filtered)
        save_summary_to_csv(
            df_nvtx_pushpop_summary,
            INFERENCE_NVTX_REPORTS_PATH,
            task,
            "nvtx_pushpop",
            category,
        )
        logger.info("Generated NVTX PushPop Summaries")

    except Exception as e:
        logger.error(f"Error in generate_nvtx_summaries: {e}")
        raise e
