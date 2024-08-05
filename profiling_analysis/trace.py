import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from profiling_analysis.helpers import (
    add_short_kernel_names,
    filter_data_between_time,
    get_start_and_end_times,
    get_unique_kernel_names_inference,
    get_reports_path,
    get_trace_path,
)
from profiling_analysis.configs.constants import (
    INFERENCE_CUDA_REPORTS_PATH,
    INFERENCE_NVTX_REPORTS_PATH,
)
from profiling_analysis import logger


def trace_process(task, base_path, operation, column_name, start_time, end_time, category=None, overwrite=False):
    try:
        # If path exists return df
        # final_path = os.path.join(base_path, "reports", f"{task}_{operation}_filtered.csv")
        final_path = get_reports_path(
            base_path, report_name=f"{task}_{operation}_trace_filtered.csv", category=category
        )
        logger.info(
            f"Task: {task}, Operation: {operation}, Column Name: {column_name}, Start Time: {start_time}, End Time : {end_time}, Final Path: {final_path}"
        )
        if not overwrite and os.path.exists(final_path):
            logger.info(
                f"Final Path Exists: {final_path}, Loading the CSV data into a DataFrame"
            )
            return pd.read_csv(final_path)

        logger.info(f"Final Path Does Not Exist: {final_path}, Processing the Trace")

        operation_trace_csv_path = get_trace_path(
            base_path, trace_name=f"{task}_{operation}_trace.csv", category=None
        )

        # Load the CSV data into a DataFrame
        df_operation_trace = pd.read_csv(operation_trace_csv_path)

        # Add index column
        df_operation_trace["index"] = np.arange(len(df_operation_trace))

        # column_name = 'Orig Start (ns)'
        df_operation_trace_filtered = filter_data_between_time(
            df_operation_trace, column_name, start_time, end_time
        )

        logger.info(f"Filtered Data: {df_operation_trace_filtered.shape}")

        df_operation_trace_filtered.to_csv(final_path, index=False)

        logger.info(f"Saved the Filtered Data to: {final_path}")

        return df_operation_trace_filtered

    except Exception as e:
        logger.error(f"Error in trace_process: {e}")
        raise e


def start_tracing_process(task="inference", category=None, overwrite=False):
    try:
        logger.info("Processing Trace after Model Loading for Inference")
        start_time, end_time = get_start_and_end_times(category)

        df_nvtx_gpu_proj_trace_filtered = trace_process(
            task,
            INFERENCE_NVTX_REPORTS_PATH,
            "nvtx_gpu_proj",
            "Orig Start (ns)",
            start_time=start_time,
            end_time=end_time,
            category=category,
            overwrite=overwrite,
        )

        df_nvtx_pushpop_trace_filtered = trace_process(
            task,
            INFERENCE_NVTX_REPORTS_PATH,
            "nvtx_pushpop",
            "Start (ns)",
            start_time=start_time,
            end_time=end_time,
            category=category,
            overwrite=overwrite,
        )

        df_cuda_api_trace_filtered = trace_process(
            task,
            INFERENCE_CUDA_REPORTS_PATH,
            "cuda_api",
            "Start (ns)",
            start_time=start_time,
            end_time=end_time,
            category=category,
            overwrite=overwrite,
        )

        df_cuda_gpu_trace_filtered = trace_process(
            task,
            INFERENCE_CUDA_REPORTS_PATH,
            "cuda_gpu",
            "Start (ns)",
            start_time=start_time,
            end_time=end_time,
            category=category,
            overwrite=overwrite,
        )

        df_cuda_kernel_exec_trace = trace_process(
            task,
            INFERENCE_CUDA_REPORTS_PATH,
            "cuda_kern_exec",
            "Kernel Start (ns)",
            start_time=start_time,
            end_time=end_time,
            category=category,
            overwrite=overwrite,
        )
        # Add short names
        # Get unique Kernel Names
        unique_kernel_names = get_unique_kernel_names_inference()
        df_cuda_kernel_exec_trace = add_short_kernel_names(
            df_cuda_kernel_exec_trace, unique_kernel_names
        )
        # Save the updated DataFrame
        df_cuda_kernel_exec_trace_path = get_reports_path(
            INFERENCE_CUDA_REPORTS_PATH,
            report_name=f"{task}_cuda_kern_exec_trace_filtered.csv",
            category=category,
        )
        df_cuda_kernel_exec_trace.to_csv(df_cuda_kernel_exec_trace_path, index=False)

        results = {
            "nvtx_gpu_proj": df_nvtx_gpu_proj_trace_filtered,
            "nvtx_pushpop": df_nvtx_pushpop_trace_filtered,
            "cuda_api": df_cuda_api_trace_filtered,
            "cuda_gpu": df_cuda_gpu_trace_filtered,
            "cuda_kernel_exec": df_cuda_kernel_exec_trace,
        }

        return results

    except Exception as e:
        logger.error(f"Error in trace_after_model_loading_inference: {e}")
        raise e

def get_processed_traces(task="inference", category=None):
    try:
        nvtx_gpu_proj_trace_filtered_path = get_reports_path(
            INFERENCE_NVTX_REPORTS_PATH,
            report_name=f"{task}_nvtx_gpu_proj_trace_filtered.csv",
            category=category,
        )
        df_nvtx_gpu_proj_trace_filtered = pd.read_csv(nvtx_gpu_proj_trace_filtered_path)
        logger.info(f"Loaded NVTX GPU Projection Trace: {df_nvtx_gpu_proj_trace_filtered.shape}, From: {nvtx_gpu_proj_trace_filtered_path}")
        
        nvtx_pushpop_trace_filtered_path = get_reports_path(
            INFERENCE_NVTX_REPORTS_PATH,
            report_name=f"{task}_nvtx_pushpop_trace_filtered.csv",
            category=category,
        )
        df_nvtx_pushpop_trace_filtered = pd.read_csv(nvtx_pushpop_trace_filtered_path)
        logger.info(f"Loaded NVTX Push Pop Trace: {df_nvtx_pushpop_trace_filtered.shape}, From: {nvtx_pushpop_trace_filtered_path}")
        
        cuda_api_trace_filtered_path = get_reports_path(
            INFERENCE_CUDA_REPORTS_PATH,
            report_name=f"{task}_cuda_api_trace_filtered.csv",
            category=category,
        )
        df_cuda_api_trace_filtered = pd.read_csv(cuda_api_trace_filtered_path)
        logger.info(f"Loaded CUDA API Trace: {df_cuda_api_trace_filtered.shape}, From: {cuda_api_trace_filtered_path}")
        
        cuda_gpu_trace_filtered_path = get_reports_path(
            INFERENCE_CUDA_REPORTS_PATH,
            report_name=f"{task}_cuda_gpu_trace_filtered.csv",
            category=category,
        )
        df_cuda_gpu_trace_filtered = pd.read_csv(cuda_gpu_trace_filtered_path)
        logger.info(f"Loaded CUDA GPU Trace: {df_cuda_gpu_trace_filtered.shape}, From: {cuda_gpu_trace_filtered_path}")
        
        df_cuda_kernel_exec_trace_path = get_reports_path(
            INFERENCE_CUDA_REPORTS_PATH,
            report_name=f"{task}_cuda_kern_exec_trace_filtered.csv",
            category=category,
        )
        df_cuda_kernel_exec_trace = pd.read_csv(df_cuda_kernel_exec_trace_path)
        logger.info(f"Loaded CUDA Kernel Execution Trace: {df_cuda_kernel_exec_trace.shape}, From: {df_cuda_kernel_exec_trace_path}")

        results = {
            "nvtx_gpu_proj": df_nvtx_gpu_proj_trace_filtered,
            "nvtx_pushpop": df_nvtx_pushpop_trace_filtered,
            "cuda_api": df_cuda_api_trace_filtered,
            "cuda_gpu": df_cuda_gpu_trace_filtered,
            "cuda_kernel_exec": df_cuda_kernel_exec_trace,
        }

        return results

    except Exception as e:
        logger.error(f"Error in trace_after_model_loading_inference: {e}")
        raise e