import os
import pandas as pd
import numpy as np
import json
import traceback, sys
from tqdm import tqdm
from profiling_analysis.logger import Logger

from profiling_analysis.helpers import add_short_kernel_names, filter_data_after_time, get_kernels_from_gpu_trace, get_unique_kernel_names_inference
from profiling_analysis.constants import (
    INFERENCE_CUDA_REPORTS_PATH,
    INFERENCE_NVTX_REPORTS_PATH,
    IGNORE_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING
)
from profiling_analysis.summary import summarize_cuda_api_operations, summarize_kernel_launch_and_exec, summarize_memory_by_size, summarize_memory_by_time

log = Logger().get_logger()



def trace_process(task, base_path, operation, column_name, ignore_time):
    try:
        # If path exists return df
        final_path = os.path.join(base_path, "reports", f"{task}_{operation}_filtered.csv")
        log.info(f"Task: {task}, Operation: {operation}, Column Name: {column_name}, Ignore Time: {ignore_time}, Final Path: {final_path}")
        if os.path.exists(final_path):
            log.info(f"Final Path Exists: {final_path}, Loading the CSV data into a DataFrame")
            return pd.read_csv(final_path)

        log.info(f"Final Path Does Not Exist: {final_path}, Processing the Trace")
        
        operation_trace_csv_path = os.path.join(base_path, f"{task}_{operation}_trace.csv")

        # Load the CSV data into a DataFrame
        df_operation_trace = pd.read_csv(operation_trace_csv_path)
        
        # Add index column
        df_operation_trace['index'] = np.arange(len(df_operation_trace))

        # column_name = 'Orig Start (ns)'
        df_operation_trace_filtered = filter_data_after_time(df_operation_trace, column_name, ignore_time)
        
        log.info(f"Filtered Data: {df_operation_trace_filtered.shape}")
        
        # csv_path = base_path + "nvtx_gpu_proj_trace_filtered.csv"
        if not os.path.exists(os.path.join(base_path, "reports")):
            os.makedirs(os.path.join(base_path, "reports"))
        
        df_operation_trace_filtered.to_csv(final_path, index=False)
        
        log.info(f"Saved the Filtered Data to: {final_path}")

        return df_operation_trace_filtered

    except Exception as e:
        log.error(f"Error in trace_process: {e}")
        raise e



def add_trace_columns(df_nvtx_gpu_proj_trace_filtered, df_cuda_gpu_trace):
    try:
        df_nvtx_gpu_proj_trace_processed = df_nvtx_gpu_proj_trace_filtered.copy()
        
        # Separate kernel and memory operations in CUDA GPU trace
        df_cuda_gpu_trace['Kernel Ops'] = df_cuda_gpu_trace.apply(lambda row: row['Name'] if pd.isnull(row['Bytes (MB)']) else None, axis=1)
        df_cuda_gpu_trace['Memory Ops'] = df_cuda_gpu_trace.apply(lambda row: row['Name'] if not pd.isnull(row['Bytes (MB)']) else None, axis=1)
        log.info("Separated kernel and memory operations in CUDA GPU trace")
        
        ## Information about CUDA APIs
        df_nvtx_gpu_proj_trace_processed['CUDA APIs'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        # Store unique CUDA API Names
        df_nvtx_gpu_proj_trace_processed['CUDA API Names'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]


        ## Information about CUDA GPU Kernels
        df_nvtx_gpu_proj_trace_processed['Kernels'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        # Store unique Kernel Names
        df_nvtx_gpu_proj_trace_processed['Kernel Names'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        # Calculate GPU execution time and idle time for each range
        df_nvtx_gpu_proj_trace_processed['GPU Execution Time'] = 0
        df_nvtx_gpu_proj_trace_processed['Kernels Match'] = ''
        df_nvtx_gpu_proj_trace_processed['Relevant Kernels from GPU Trace'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        log.info("Added columns for GPU Kernels")

        ## Information about CUDA GPU Memory Operations
        df_nvtx_gpu_proj_trace_processed['Memory Ops'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        # Store unique Memory Operation Names
        df_nvtx_gpu_proj_trace_processed['Memory Operation Names'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        log.info("Added columns for GPU Memory Operations")
        
        ## Store CorrIDs for Kernels and Memory Operations
        df_nvtx_gpu_proj_trace_filtered['GPU Kernel CorrIDs'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_filtered))]
        df_nvtx_gpu_proj_trace_filtered['GPU Memory CorrIDs'] = [[] for _ in range(len(df_nvtx_gpu_proj_trace_filtered))]
        log.info("Added columns for CorrIDs")
        
        ## Information about CUDA API to Kernel Mapping
        df_nvtx_gpu_proj_trace_processed['CUDA API to Kernel Mapping'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        df_nvtx_gpu_proj_trace_processed['CUDA API to Memory Mapping'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        log.info("Added columns for CUDA API to Kernel Mapping")
        
        # Columns for Summary
        # Summarize operations for the range
        df_nvtx_gpu_proj_trace_processed['CUDA API Summary'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        df_nvtx_gpu_proj_trace_processed['Memory Summary by Size'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        df_nvtx_gpu_proj_trace_processed['Memory Summary by Time'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        df_nvtx_gpu_proj_trace_processed['Kernel Summary'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        df_nvtx_gpu_proj_trace_processed['Kernel Summary (Short Names)'] = [{} for _ in range(len(df_nvtx_gpu_proj_trace_processed))]
        log.info("Added columns for Summary")
        
        return df_nvtx_gpu_proj_trace_processed
    
    except Exception as e:
        log.error(f"Error in add_trace_columns: {e}")
        raise e


def get_relevant_cuda_apis(df_cuda_api_trace, range_start, range_end):
    try:
        # Filter relevant CUDA API calls
        relevant_cuda_apis = df_cuda_api_trace[(df_cuda_api_trace['Start (ns)'] >= range_start) & 
                                            #    (df_cuda_api_trace['Start (ns)'] + df_cuda_api_trace['Duration (ns)'] <= range_end)]
                                                (df_cuda_api_trace['Start (ns)'] <= range_end)]
                                                #  (df_cuda_api_trace['Start (ns)'] <= range_end) &
                                                #     (~df_cuda_api_trace['Name'].isin(irrelevant_cuda_apis))]
        relevant_cuda_apis = relevant_cuda_apis.drop_duplicates(subset=['CorrID', 'Name', 'Start (ns)'])
        
        # Unique CUDA API names
        unique_cuda_api_names = relevant_cuda_apis['Name'].unique()
        
        return relevant_cuda_apis, unique_cuda_api_names

    except Exception as e:
        log.error(f"Error in get_relevant_cuda_apis: {e}")
        raise e

    
def get_relevant_kernels(df_cuda_kernel_exec_trace, range_start, range_end):
    try:
        # Filter relevant CUDA kernels
        relevant_kernels = df_cuda_kernel_exec_trace[(df_cuda_kernel_exec_trace['Kernel Start (ns)'] >= range_start) & 
                                                    #  (df_cuda_kernel_exec_trace['Kernel Start (ns)'] + df_cuda_kernel_exec_trace['Kernel Dur'] <= range_end)]
                                                        (df_cuda_kernel_exec_trace['Kernel Start (ns)'] <= range_end)]
        relevant_kernels = relevant_kernels.drop_duplicates(subset=['Kernel Start (ns)', 'Kernel Name'])
        
        # Unique Kernel names
        unique_kernel_names = relevant_kernels['Kernel Name'].unique()
        
        return relevant_kernels, unique_kernel_names
    
    except Exception as e:
        log.error(f"Error in get_relevant_kernels: {e}")
        raise e


def get_gpu_execution_time(relevant_kernels):
    try:
        return relevant_kernels['Kernel Dur (ns)'].sum()
    
    except Exception as e:
        log.error(f"Error in get_gpu_execution_time: {e}")
        raise e    


def get_relevant_kernels_from_gpu_trace(df_cuda_gpu_trace, range_start, range_end):
    try:
        # Filter relevant kernels
        relevant_kernels_from_gpu_trace = get_kernels_from_gpu_trace(df_cuda_gpu_trace, range_start, range_end)
        relevant_kernels_from_gpu_trace = relevant_kernels_from_gpu_trace.drop_duplicates(subset=['Start (ns)', 'Name'])
        
        return relevant_kernels_from_gpu_trace

    except Exception as e:
        log.error(f"Error in get_relevant_kernels_from_gpu_trace: {e}")
        raise e
    
    
def get_relevant_memory_ops(df_cuda_gpu_trace, range_start, range_end):
    try:
        # Filter relevant memory operations
        relevant_memory_ops = df_cuda_gpu_trace[(df_cuda_gpu_trace['Start (ns)'] >= range_start) & 
                                                (df_cuda_gpu_trace['Start (ns)'] + df_cuda_gpu_trace['Duration (ns)'] <= range_end) & 
                                                (~df_cuda_gpu_trace['Bytes (MB)'].isnull())]
        relevant_memory_ops = relevant_memory_ops.drop_duplicates(subset=['Start (ns)', 'Name'])
        
        # Unique Memory Operation names
        unique_memory_op_names = relevant_memory_ops['Name'].unique()
        
        return relevant_memory_ops, unique_memory_op_names

    except Exception as e:
        log.error(f"Error in get_relevant_memory_ops: {e}")
        raise e
    
    
def get_api_mapping(df_cuda_gpu_trace, relevant_cuda_apis, relevant_kernels_from_gpu_trace, relevant_memory_ops):
    try:
        # Create a mapping between CUDA API calls and kernels based on CorrId within the range
        # Create a mapping between CUDA API calls and memory operations based on CorrId within the range
        api_kernel_mapping = []
        api_memory_mapping = []

        for _, api_row in relevant_cuda_apis.iterrows():
            api_corr_id = api_row['CorrID']
            api_name = api_row['Name']
            # If CorrId is in the relevant kernels, get the kernel names
            if api_corr_id in relevant_kernels_from_gpu_trace['CorrId'].values:
                mapped_kernels = df_cuda_gpu_trace[df_cuda_gpu_trace['CorrId'] == api_corr_id]['Name'].tolist()
                api_kernel_mapping.append({
                    'CorrID': api_corr_id,
                    'CUDA API': api_name,
                    'Kernels': mapped_kernels
                })
            # If CorrId is in the relevant memory operations, get the memory operation names
            if api_corr_id in relevant_memory_ops['CorrId'].values:
                mapped_memory_ops = df_cuda_gpu_trace[df_cuda_gpu_trace['CorrId'] == api_corr_id]['Name'].tolist()
                api_memory_mapping.append({
                    'CorrID': api_corr_id,
                    'CUDA API': api_name,
                    'Memory Operations': mapped_memory_ops
                })
        
        return api_kernel_mapping, api_memory_mapping
    
    except Exception as e:
        log.error(f"Error in get_api_mapping: {e}")
        raise e
    
    

def get_corr_ids(relevant_kernels_from_gpu_trace, relevant_memory_ops):
    try:
        # # GPU CorrIDs
        if relevant_kernels_from_gpu_trace.empty:
            gpu_kernel_corr_ids = ''
        else:
            gpu_kernel_corr_ids = relevant_kernels_from_gpu_trace['CorrId'].unique()
            # Store as strings with comma separated values
            gpu_kernel_corr_ids = ','.join(map(str, gpu_kernel_corr_ids))
            
        # GPU Memory CorrIDs
        if relevant_memory_ops.empty:
            gpu_memory_corr_ids = ''
        else:
            gpu_memory_corr_ids = relevant_memory_ops['CorrId'].unique()
            # Store as strings with comma separated values
            gpu_memory_corr_ids = ','.join(map(str, gpu_memory_corr_ids))
            
        return gpu_kernel_corr_ids, gpu_memory_corr_ids

    except Exception as e:
        log.error(f"Error in get_corr_ids: {e}")
        raise e

       
def gpu_and_cpu_times(df_nvtx_gpu_proj_trace_filtered, df_cuda_api_trace, df_cuda_gpu_trace, df_cuda_kernel_exec_trace):
    try:
        log.info("Processing GPU and CPU Times")
        
        df_nvtx_gpu_proj_trace_processed = add_trace_columns(df_nvtx_gpu_proj_trace_filtered, df_cuda_gpu_trace)
        
        # Loop through the NVTX GPU projection dataframe with a progress bar
        for idx, row in tqdm(df_nvtx_gpu_proj_trace_processed.iterrows(), total=len(df_nvtx_gpu_proj_trace_processed), desc="Processing NVTX GPU Projection"):
            range_start = row['Projected Start (ns)']
            range_end = range_start + row['Projected Duration (ns)']
            
            relevant_cuda_apis, unique_cuda_api_names = get_relevant_cuda_apis(df_cuda_api_trace, range_start, range_end)
            df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA APIs'] = relevant_cuda_apis.to_dict('records')
            log.info(f"Index: {idx}, Relevant CUDA APIs added to df_nvtx_gpu_proj_trace_processed")
            
            # Store as strings with comma separated values
            df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API Names'] = ','.join(unique_cuda_api_names)
            log.info(f"Index: {idx}, Unique CUDA API Names added to df_nvtx_gpu_proj_trace_processed")
            
            # Filter relevant CUDA kernels
            relevant_kernels, unique_kernel_names = get_relevant_kernels(df_cuda_kernel_exec_trace, range_start, range_end)
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernels'] = relevant_kernels.to_dict('records')
            log.info(f"Index: {idx}, Relevant Kernels added to df_nvtx_gpu_proj_trace_processed")
            
            # Store as strings with comma separated values
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernel Names'] = ','.join(unique_kernel_names)
            log.info(f"Index: {idx}, Unique Kernel Names added to df_nvtx_gpu_proj_trace_processed")
            
            
            # Sum of kernel durations
            df_nvtx_gpu_proj_trace_processed.at[idx, 'GPU Execution Time'] = get_gpu_execution_time(relevant_kernels)
            
            # Get kernels from df_cuda_gpu_trace
            relevant_kernels_from_gpu_trace = get_relevant_kernels_from_gpu_trace(df_cuda_gpu_trace, range_start, range_end)
            log.info(f"Index: {idx}, Relevant Kernels from GPU Trace added to df_nvtx_gpu_proj_trace_processed")
            
            # Check if the kernels match
            kernels_match = set(relevant_kernels['Kernel Start (ns)']) == set(relevant_kernels_from_gpu_trace['Start (ns)'])
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernels Match'] = 'Yes' if kernels_match else 'No'
            log.info(f"Index: {idx}, Kernels Match: {kernels_match}")
            
            # Store relevant kernels from GPU trace only if they don't match
            if not kernels_match:
                log.info(f"Index: {idx}, Kernels do not match, saving relevant kernels from GPU trace")
                df_nvtx_gpu_proj_trace_processed.at[idx, 'Relevant Kernels from GPU Trace'] = relevant_kernels_from_gpu_trace.to_dict('records')
            
            # Filter relevant memory operations
            relevant_memory_ops, unique_memory_op_names = get_relevant_memory_ops(df_cuda_gpu_trace, range_start, range_end)
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Ops'] = relevant_memory_ops.to_dict('records')
            log.info(f"Index: {idx}, Relevant Memory Operations added to df_nvtx_gpu_proj_trace_processed")
            # Store as strings with comma separated values
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Operation Names'] = ','.join(unique_memory_op_names)
            log.info(f"Index: {idx}, Unique Memory Operation Names added to df_nvtx_gpu_proj_trace_processed")
            
            # Get API Mapping
            api_kernel_mapping, api_memory_mapping = get_api_mapping(df_cuda_gpu_trace, relevant_cuda_apis, relevant_kernels_from_gpu_trace, relevant_memory_ops)
            df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API to Kernel Mapping'] = api_kernel_mapping    
            df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API to Memory Mapping'] = api_memory_mapping
            
            # Get CorrIDs
            gpu_kernel_corr_ids, gpu_memory_corr_ids = get_corr_ids(relevant_kernels_from_gpu_trace, relevant_memory_ops)
            df_nvtx_gpu_proj_trace_processed.at[idx, 'GPU Kernel CorrIDs'] = gpu_kernel_corr_ids
            df_nvtx_gpu_proj_trace_processed.at[idx, 'GPU Memory CorrIDs'] = gpu_memory_corr_ids
            
            # Summarize operations for the range
            df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API Summary'] = summarize_cuda_api_operations(relevant_cuda_apis).to_dict('records')
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Summary by Size'] = summarize_memory_by_size(relevant_memory_ops).to_dict('records')
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Summary by Time'] = summarize_memory_by_time(relevant_memory_ops).to_dict('records')
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernel Summary'] = summarize_kernel_launch_and_exec(relevant_kernels).to_dict('records')
            df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernel Summary (Short Names)'] = summarize_kernel_launch_and_exec(relevant_kernels, short_kernel_name=True).to_dict('records')
            
            # Save if idx is 100 or 1000
            if idx == 100 or idx == 1000 or idx == 10000:
                csv_path = os.path.join(INFERENCE_NVTX_REPORTS_PATH, f"test_nvtx_gpu_proj_trace_processed_{idx}.csv")
                df_nvtx_gpu_proj_trace_processed.head(idx).to_csv(csv_path, index=False)
                
        # Save processed data
        csv_path = os.path.join(INFERENCE_NVTX_REPORTS_PATH, "nvtx_gpu_proj_trace_processed.csv")
        df_nvtx_gpu_proj_trace_processed.to_csv(csv_path, index=False)
        
        return df_nvtx_gpu_proj_trace_processed
    
    except Exception as e:
        log.error(f"Error in gpu_and_cpu_times: {e}")
        raise e
    
    
    
def trace_after_model_loading_inference():
    try:
        task = "inference"
        
        log.info("Processing Trace after Model Loading for Inference")
        
        df_nvtx_gpu_proj_trace_filtered = trace_process(task, INFERENCE_NVTX_REPORTS_PATH, "nvtx_gpu_proj", "Orig Start (ns)", IGNORE_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING)

        df_nvtx_pushpop_trace_filtered = trace_process(task, INFERENCE_NVTX_REPORTS_PATH, "nvtx_pushpop", "Start (ns)", IGNORE_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING)
        
        df_cuda_api_trace_filtered = trace_process(task, INFERENCE_CUDA_REPORTS_PATH, "cuda_api", "Start (ns)", IGNORE_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING)
        
        df_cuda_gpu_trace_filtered = trace_process(task, INFERENCE_CUDA_REPORTS_PATH, "cuda_gpu", "Start (ns)", IGNORE_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING)
        
        # Get unique Kernel Names
        unique_kernel_names = get_unique_kernel_names_inference()
        df_cuda_kernel_exec_trace = trace_process(task, INFERENCE_CUDA_REPORTS_PATH, "cuda_kern_exec", "Kernel Start (ns)", IGNORE_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING)
        # Add short names
        df_cuda_kernel_exec_trace = add_short_kernel_names(df_cuda_kernel_exec_trace, unique_kernel_names)
        
        df_nvtx_gpu_proj_trace_processed = gpu_and_cpu_times(df_nvtx_gpu_proj_trace_filtered, df_cuda_api_trace_filtered, df_cuda_gpu_trace_filtered, df_cuda_kernel_exec_trace)
        
        results = {
            "nvtx_gpu_proj": df_nvtx_gpu_proj_trace_filtered,
            "nvtx_gpu_proj_processed" : df_nvtx_gpu_proj_trace_processed,
            "nvtx_pushpop": df_nvtx_pushpop_trace_filtered,
            "cuda_api": df_cuda_api_trace_filtered,
            "cuda_gpu": df_cuda_gpu_trace_filtered,
            "cuda_kernel_exec": df_cuda_kernel_exec_trace
        }
        
        return results
    
    except Exception as e:
        log.error(f"Error in trace_after_model_loading_inference: {e}")
        raise e
