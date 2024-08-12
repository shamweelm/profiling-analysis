import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from profiling_analysis.summary import (
    summarize_cuda_api_operations,
    summarize_kernel_launch_and_exec,
    summarize_memory_by_size,
    summarize_memory_by_time,
)
from profiling_analysis.helpers import (
    get_kernels_from_gpu_trace,
    get_reports_path,
)
from profiling_analysis.configs.constants import (
    INFERENCE_NVTX_REPORTS_PATH,
    INFERENCE_CUDA_REPORTS_PATH
)
from profiling_analysis.trace import get_processed_traces, start_tracing_process
from profiling_analysis import logger


def add_trace_columns(df_nvtx_gpu_proj_trace_filtered, df_cuda_gpu_trace):
    try:
        df_nvtx_gpu_proj_trace_processed = df_nvtx_gpu_proj_trace_filtered.copy()

        # Separate kernel and memory operations in CUDA GPU trace
        df_cuda_gpu_trace["Kernel Ops"] = df_cuda_gpu_trace.apply(
            lambda row: row["Name"] if pd.isnull(row["Bytes (MB)"]) else None, axis=1
        )
        df_cuda_gpu_trace["Memory Ops"] = df_cuda_gpu_trace.apply(
            lambda row: row["Name"] if not pd.isnull(row["Bytes (MB)"]) else None,
            axis=1,
        )
        logger.info("Separated kernel and memory operations in CUDA GPU trace")

        ## Information about CUDA APIs
        df_nvtx_gpu_proj_trace_processed["CUDA APIs"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        # Store unique CUDA API Names
        df_nvtx_gpu_proj_trace_processed["CUDA API Names"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]

        ## Information about CUDA GPU Kernels
        df_nvtx_gpu_proj_trace_processed["Kernels"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        # Store unique Kernel Names
        df_nvtx_gpu_proj_trace_processed["Kernel Names"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        # Calculate GPU execution time and idle time for each range
        df_nvtx_gpu_proj_trace_processed["GPU Execution Time"] = 0
        df_nvtx_gpu_proj_trace_processed["Kernels Match"] = ""
        df_nvtx_gpu_proj_trace_processed["Relevant Kernels from GPU Trace"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        logger.info("Added columns for GPU Kernels")

        ## Information about CUDA GPU Memory Operations
        df_nvtx_gpu_proj_trace_processed["Memory Ops"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        # Store unique Memory Operation Names
        df_nvtx_gpu_proj_trace_processed["Memory Operation Names"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        logger.info("Added columns for GPU Memory Operations")

        ## Store CorrIDs for Kernels and Memory Operations
        df_nvtx_gpu_proj_trace_filtered["GPU Kernel CorrIDs"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_filtered))
        ]
        df_nvtx_gpu_proj_trace_filtered["GPU Memory CorrIDs"] = [
            [] for _ in range(len(df_nvtx_gpu_proj_trace_filtered))
        ]
        logger.info("Added columns for CorrIDs")

        ## Information about CUDA API to Kernel Mapping
        df_nvtx_gpu_proj_trace_processed["CUDA API to Kernel Mapping"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        df_nvtx_gpu_proj_trace_processed["CUDA API to Memory Mapping"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        logger.info("Added columns for CUDA API to Kernel Mapping")

        # Columns for Summary
        # Summarize operations for the range
        df_nvtx_gpu_proj_trace_processed["CUDA API Summary"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        df_nvtx_gpu_proj_trace_processed["Memory Summary by Size"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        df_nvtx_gpu_proj_trace_processed["Memory Summary by Time"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        df_nvtx_gpu_proj_trace_processed["Kernel Summary"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        df_nvtx_gpu_proj_trace_processed["Kernel Summary (Short Names)"] = [
            {} for _ in range(len(df_nvtx_gpu_proj_trace_processed))
        ]
        logger.info("Added columns for Summary")

        return df_nvtx_gpu_proj_trace_processed

    except Exception as e:
        logger.error(f"Error in add_trace_columns: {e}")
        raise e


def get_relevant_cuda_apis(df_cuda_api_trace, range_start, range_end):
    try:
        # Filter relevant CUDA API calls
        relevant_cuda_apis = df_cuda_api_trace[
            (df_cuda_api_trace["Start (ns)"] >= range_start)
            &
            #    (df_cuda_api_trace['Start (ns)'] + df_cuda_api_trace['Duration (ns)'] <= range_end)]
            (df_cuda_api_trace["Start (ns)"] <= range_end)
        ]
        #  (df_cuda_api_trace['Start (ns)'] <= range_end) &
        #     (~df_cuda_api_trace['Name'].isin(irrelevant_cuda_apis))]
        relevant_cuda_apis = relevant_cuda_apis.drop_duplicates(
            subset=["CorrID", "Name", "Start (ns)"]
        )

        # Unique CUDA API names
        unique_cuda_api_names = relevant_cuda_apis["Name"].unique()

        return relevant_cuda_apis, unique_cuda_api_names

    except Exception as e:
        logger.error(f"Error in get_relevant_cuda_apis: {e}")
        raise e


def get_relevant_kernels(df_cuda_kernel_exec_trace, range_start, range_end):
    try:
        if (df_cuda_kernel_exec_trace is None) or (df_cuda_kernel_exec_trace.empty):
            return pd.DataFrame(), np.array([])
        
        # Filter relevant CUDA kernels
        relevant_kernels = df_cuda_kernel_exec_trace[
            (df_cuda_kernel_exec_trace["Kernel Start (ns)"] >= range_start)
            &
            #  (df_cuda_kernel_exec_trace['Kernel Start (ns)'] + df_cuda_kernel_exec_trace['Kernel Dur'] <= range_end)]
            (df_cuda_kernel_exec_trace["Kernel Start (ns)"] <= range_end)
        ]
        relevant_kernels = relevant_kernels.drop_duplicates(
            subset=["Kernel Start (ns)", "Kernel Name"]
        )

        # Unique Kernel names
        unique_kernel_names = relevant_kernels["Kernel Name"].unique()

        return relevant_kernels, unique_kernel_names

    except Exception as e:
        logger.error(f"Error in get_relevant_kernels: {e}")
        raise e


def get_gpu_execution_time(relevant_kernels):
    try:
        if relevant_kernels.empty or relevant_kernels is None:
            return 0
        return relevant_kernels["Kernel Dur (ns)"].sum()

    except Exception as e:
        logger.error(f"Error in get_gpu_execution_time: {e}")
        raise e


def get_relevant_kernels_from_gpu_trace(df_cuda_gpu_trace, range_start, range_end):
    try:
        # Filter relevant kernels
        relevant_kernels_from_gpu_trace = get_kernels_from_gpu_trace(
            df_cuda_gpu_trace, range_start, range_end
        )
        relevant_kernels_from_gpu_trace = (
            relevant_kernels_from_gpu_trace.drop_duplicates(
                subset=["Start (ns)", "Name"]
            )
        )

        return relevant_kernels_from_gpu_trace

    except Exception as e:
        logger.error(f"Error in get_relevant_kernels_from_gpu_trace: {e}")
        raise e


def get_relevant_memory_ops(df_cuda_gpu_trace, range_start, range_end):
    try:
        # Filter relevant memory operations
        relevant_memory_ops = df_cuda_gpu_trace[
            (df_cuda_gpu_trace["Start (ns)"] >= range_start)
            & (
                df_cuda_gpu_trace["Start (ns)"] + df_cuda_gpu_trace["Duration (ns)"]
                <= range_end
            )
            & (~df_cuda_gpu_trace["Bytes (MB)"].isnull())
        ]
        relevant_memory_ops = relevant_memory_ops.drop_duplicates(
            subset=["Start (ns)", "Name"]
        )

        # Unique Memory Operation names
        unique_memory_op_names = relevant_memory_ops["Name"].unique()

        return relevant_memory_ops, unique_memory_op_names

    except Exception as e:
        logger.error(f"Error in get_relevant_memory_ops: {e}")
        raise e


def get_api_mapping(
    df_cuda_gpu_trace,
    relevant_cuda_apis,
    relevant_kernels_from_gpu_trace,
    relevant_memory_ops,
):
    try:
        # Create a mapping between CUDA API calls and kernels based on CorrId within the range
        # Create a mapping between CUDA API calls and memory operations based on CorrId within the range
        api_kernel_mapping = []
        api_memory_mapping = []

        for _, api_row in relevant_cuda_apis.iterrows():
            api_corr_id = api_row["CorrID"]
            api_name = api_row["Name"]
            # If CorrId is in the relevant kernels, get the kernel names
            if api_corr_id in relevant_kernels_from_gpu_trace["CorrId"].values:
                mapped_kernels = df_cuda_gpu_trace[
                    df_cuda_gpu_trace["CorrId"] == api_corr_id
                ]["Name"].tolist()
                api_kernel_mapping.append(
                    {
                        "CorrID": api_corr_id,
                        "CUDA API": api_name,
                        "Kernels": mapped_kernels,
                    }
                )
            # If CorrId is in the relevant memory operations, get the memory operation names
            if api_corr_id in relevant_memory_ops["CorrId"].values:
                mapped_memory_ops = df_cuda_gpu_trace[
                    df_cuda_gpu_trace["CorrId"] == api_corr_id
                ]["Name"].tolist()
                api_memory_mapping.append(
                    {
                        "CorrID": api_corr_id,
                        "CUDA API": api_name,
                        "Memory Operations": mapped_memory_ops,
                    }
                )

        return api_kernel_mapping, api_memory_mapping

    except Exception as e:
        logger.error(f"Error in get_api_mapping: {e}")
        raise e


def get_corr_ids(relevant_kernels_from_gpu_trace, relevant_memory_ops):
    try:
        # # GPU CorrIDs
        if relevant_kernels_from_gpu_trace.empty:
            gpu_kernel_corr_ids = ""
        else:
            gpu_kernel_corr_ids = relevant_kernels_from_gpu_trace["CorrId"].unique()
            # Store as strings with comma separated values
            gpu_kernel_corr_ids = ",".join(map(str, gpu_kernel_corr_ids))

        # GPU Memory CorrIDs
        if relevant_memory_ops.empty:
            gpu_memory_corr_ids = ""
        else:
            gpu_memory_corr_ids = relevant_memory_ops["CorrId"].unique()
            # Store as strings with comma separated values
            gpu_memory_corr_ids = ",".join(map(str, gpu_memory_corr_ids))

        return gpu_kernel_corr_ids, gpu_memory_corr_ids

    except Exception as e:
        logger.error(f"Error in get_corr_ids: {e}")
        raise e


# def gpu_and_cpu_times(df_nvtx_gpu_proj_trace_filtered, df_cuda_api_trace, df_cuda_gpu_trace, df_cuda_kernel_exec_trace):
#     try:
#         logger.info("Processing GPU and CPU Times")

#         df_nvtx_gpu_proj_trace_processed = add_trace_columns(df_nvtx_gpu_proj_trace_filtered, df_cuda_gpu_trace)

#         # Loop through the NVTX GPU projection dataframe with a progress bar
#         for idx, row in tqdm(df_nvtx_gpu_proj_trace_processed.iterrows(), total=len(df_nvtx_gpu_proj_trace_processed), desc="Processing NVTX GPU Projection"):
#             range_start = row['Projected Start (ns)']
#             range_end = range_start + row['Projected Duration (ns)']

#             relevant_cuda_apis, unique_cuda_api_names = get_relevant_cuda_apis(df_cuda_api_trace, range_start, range_end)
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA APIs'] = relevant_cuda_apis.to_dict('records')
#             logger.info(f"Index: {idx}, Relevant CUDA APIs added to df_nvtx_gpu_proj_trace_processed")

#             # Store as strings with comma separated values
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API Names'] = ','.join(unique_cuda_api_names)
#             logger.info(f"Index: {idx}, Unique CUDA API Names added to df_nvtx_gpu_proj_trace_processed")

#             # Filter relevant CUDA kernels
#             relevant_kernels, unique_kernel_names = get_relevant_kernels(df_cuda_kernel_exec_trace, range_start, range_end)
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernels'] = relevant_kernels.to_dict('records')
#             logger.info(f"Index: {idx}, Relevant Kernels added to df_nvtx_gpu_proj_trace_processed")

#             # Store as strings with comma separated values
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernel Names'] = ','.join(unique_kernel_names)
#             logger.info(f"Index: {idx}, Unique Kernel Names added to df_nvtx_gpu_proj_trace_processed")


#             # Sum of kernel durations
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'GPU Execution Time'] = get_gpu_execution_time(relevant_kernels)

#             # Get kernels from df_cuda_gpu_trace
#             relevant_kernels_from_gpu_trace = get_relevant_kernels_from_gpu_trace(df_cuda_gpu_trace, range_start, range_end)
#             logger.info(f"Index: {idx}, Relevant Kernels from GPU Trace added to df_nvtx_gpu_proj_trace_processed")

#             # Check if the kernels match
#             kernels_match = set(relevant_kernels['Kernel Start (ns)']) == set(relevant_kernels_from_gpu_trace['Start (ns)'])
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernels Match'] = 'Yes' if kernels_match else 'No'
#             logger.info(f"Index: {idx}, Kernels Match: {kernels_match}")

#             # Store relevant kernels from GPU trace only if they don't match
#             if not kernels_match:
#                 logger.info(f"Index: {idx}, Kernels do not match, saving relevant kernels from GPU trace")
#                 df_nvtx_gpu_proj_trace_processed.at[idx, 'Relevant Kernels from GPU Trace'] = relevant_kernels_from_gpu_trace.to_dict('records')

#             # Filter relevant memory operations
#             relevant_memory_ops, unique_memory_op_names = get_relevant_memory_ops(df_cuda_gpu_trace, range_start, range_end)
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Ops'] = relevant_memory_ops.to_dict('records')
#             logger.info(f"Index: {idx}, Relevant Memory Operations added to df_nvtx_gpu_proj_trace_processed")
#             # Store as strings with comma separated values
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Operation Names'] = ','.join(unique_memory_op_names)
#             logger.info(f"Index: {idx}, Unique Memory Operation Names added to df_nvtx_gpu_proj_trace_processed")

#             # Get API Mapping
#             api_kernel_mapping, api_memory_mapping = get_api_mapping(df_cuda_gpu_trace, relevant_cuda_apis, relevant_kernels_from_gpu_trace, relevant_memory_ops)
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API to Kernel Mapping'] = api_kernel_mapping
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API to Memory Mapping'] = api_memory_mapping

#             # Get CorrIDs
#             gpu_kernel_corr_ids, gpu_memory_corr_ids = get_corr_ids(relevant_kernels_from_gpu_trace, relevant_memory_ops)
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'GPU Kernel CorrIDs'] = gpu_kernel_corr_ids
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'GPU Memory CorrIDs'] = gpu_memory_corr_ids

#             # Summarize operations for the range
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'CUDA API Summary'] = summarize_cuda_api_operations(relevant_cuda_apis).to_dict('records')
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Summary by Size'] = summarize_memory_by_size(relevant_memory_ops).to_dict('records')
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Memory Summary by Time'] = summarize_memory_by_time(relevant_memory_ops).to_dict('records')
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernel Summary'] = summarize_kernel_launch_and_exec(relevant_kernels).to_dict('records')
#             df_nvtx_gpu_proj_trace_processed.at[idx, 'Kernel Summary (Short Names)'] = summarize_kernel_launch_and_exec(relevant_kernels, short_kernel_name=True).to_dict('records')

#             # Save if idx is 100 or 1000
#             if idx == 100 or idx == 1000 or idx == 10000:
#                 csv_path = os.path.join(INFERENCE_NVTX_REPORTS_PATH, f"test_nvtx_gpu_proj_trace_processed_{idx}.csv")
#                 df_nvtx_gpu_proj_trace_processed.head(idx).to_csv(csv_path, index=False)

#         # Save processed data
#         csv_path = os.path.join(INFERENCE_NVTX_REPORTS_PATH, "nvtx_gpu_proj_trace_processed.csv")
#         df_nvtx_gpu_proj_trace_processed.to_csv(csv_path, index=False)

#         return df_nvtx_gpu_proj_trace_processed

#     except Exception as e:
#         logger.error(f"Error in gpu_and_cpu_times: {e}")
#         raise e


def process_chunk(
    chunk, df_cuda_api_trace, df_cuda_gpu_trace, df_cuda_kernel_exec_trace
):
    try:
        chunk_processed = add_trace_columns(chunk, df_cuda_gpu_trace)
        chunk_processed.reset_index(drop=True, inplace=True)

        for idx in tqdm(
            range(len(chunk_processed)), desc="Processing rows in chunk", leave=False
        ):
            row = chunk_processed.iloc[idx]
            range_start = row["Projected Start (ns)"]
            range_end = range_start + row["Projected Duration (ns)"]

            relevant_cuda_apis, unique_cuda_api_names = get_relevant_cuda_apis(
                df_cuda_api_trace, range_start, range_end
            )
            chunk_processed.at[idx, "CUDA APIs"] = relevant_cuda_apis.to_dict("records")
            chunk_processed.at[idx, "CUDA API Names"] = ",".join(unique_cuda_api_names)

            relevant_kernels, unique_kernel_names = get_relevant_kernels(
                df_cuda_kernel_exec_trace, range_start, range_end
            )
            chunk_processed.at[idx, "Kernels"] = relevant_kernels.to_dict("records")
            chunk_processed.at[idx, "Kernel Names"] = ",".join(unique_kernel_names)

            chunk_processed.at[idx, "GPU Execution Time"] = get_gpu_execution_time(
                relevant_kernels
            )

            relevant_kernels_from_gpu_trace = get_relevant_kernels_from_gpu_trace(
                df_cuda_gpu_trace, range_start, range_end
            )
            if relevant_kernels_from_gpu_trace.empty:
                kernels_match = False
            else:
                kernels_match = set(relevant_kernels["Kernel Start (ns)"]) == set(
                    relevant_kernels_from_gpu_trace["Start (ns)"]
                )
            chunk_processed.at[idx, "Kernels Match"] = "Yes" if kernels_match else "No"

            if not kernels_match:
                chunk_processed.at[idx, "Relevant Kernels from GPU Trace"] = (
                    relevant_kernels_from_gpu_trace.to_dict("records")
                )

            relevant_memory_ops, unique_memory_op_names = get_relevant_memory_ops(
                df_cuda_gpu_trace, range_start, range_end
            )
            chunk_processed.at[idx, "Memory Ops"] = relevant_memory_ops.to_dict(
                "records"
            )
            chunk_processed.at[idx, "Memory Operation Names"] = ",".join(
                unique_memory_op_names
            )

            api_kernel_mapping, api_memory_mapping = get_api_mapping(
                df_cuda_gpu_trace,
                relevant_cuda_apis,
                relevant_kernels_from_gpu_trace,
                relevant_memory_ops,
            )
            chunk_processed.at[idx, "CUDA API to Kernel Mapping"] = api_kernel_mapping
            chunk_processed.at[idx, "CUDA API to Memory Mapping"] = api_memory_mapping

            gpu_kernel_corr_ids, gpu_memory_corr_ids = get_corr_ids(
                relevant_kernels_from_gpu_trace, relevant_memory_ops
            )
            chunk_processed.at[idx, "GPU Kernel CorrIDs"] = gpu_kernel_corr_ids
            chunk_processed.at[idx, "GPU Memory CorrIDs"] = gpu_memory_corr_ids

            chunk_processed.at[idx, "CUDA API Summary"] = summarize_cuda_api_operations(
                relevant_cuda_apis
            ).to_dict("records")
            chunk_processed.at[idx, "Memory Summary by Size"] = (
                summarize_memory_by_size(relevant_memory_ops).to_dict("records")
            )
            chunk_processed.at[idx, "Memory Summary by Time"] = (
                summarize_memory_by_time(relevant_memory_ops).to_dict("records")
            )
            chunk_processed.at[idx, "Kernel Summary"] = (
                summarize_kernel_launch_and_exec(relevant_kernels).to_dict("records")
            )
            chunk_processed.at[idx, "Kernel Summary (Short Names)"] = (
                summarize_kernel_launch_and_exec(
                    relevant_kernels, short_kernel_name=True
                ).to_dict("records")
            )

        return chunk_processed

    except Exception as e:
        logger.error(f"Error in process_chunk: {e}")
        raise e


def process_chunk_wrapper(args):
    return process_chunk(*args)


def gpu_and_cpu_times(
    df_nvtx_gpu_proj_trace_filtered,
    df_cuda_api_trace,
    df_cuda_gpu_trace,
    df_cuda_kernel_exec_trace,
    category=None,
):
    try:
        logger.info("Processing GPU and CPU Times")

        num_chunks = mp.cpu_count()
        # chunk_size = len(df_nvtx_gpu_proj_trace_filtered) // num_chunks
        chunk_size = max(len(df_nvtx_gpu_proj_trace_filtered) // num_chunks, 1)
        
        chunks = [
            df_nvtx_gpu_proj_trace_filtered.iloc[i : i + chunk_size]
            for i in range(0, len(df_nvtx_gpu_proj_trace_filtered), chunk_size)
        ]
        logger.info(f"Split data into {num_chunks} chunks")

        with mp.Pool(num_chunks) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(
                    process_chunk_wrapper,
                    [
                        (
                            chunk,
                            df_cuda_api_trace,
                            df_cuda_gpu_trace,
                            df_cuda_kernel_exec_trace,
                        )
                        for chunk in chunks
                    ],
                ),
                desc="Processing chunks",
                total=len(chunks),
            ):
                results.append(result)
        
        logger.info("Processed all chunks")

        print("results: ", results)
        df_nvtx_gpu_proj_trace_processed = pd.concat(results)

        csv_path = get_reports_path(
            INFERENCE_NVTX_REPORTS_PATH, report_name="nvtx_gpu_proj_trace_processed.csv", category=category
        )
        df_nvtx_gpu_proj_trace_processed.to_csv(csv_path, index=False)
        logger.info(f"Saved processed data to: {csv_path}")

        return df_nvtx_gpu_proj_trace_processed

    except Exception as e:
        logger.error(f"Error in gpu_and_cpu_times: {e}")
        raise e
    

def start_gpu_cpu_times_process(task="inference", category=None, results={}, sample_size=None):
    try:
        if not results:
            raise ValueError("Results dictionary is empty")
        
        df_nvtx_gpu_proj_trace_filtered = results["nvtx_gpu_proj"]
        df_nvtx_pushpop_trace_filtered = results["nvtx_pushpop"]
        df_cuda_api_trace_filtered = results["cuda_api"]
        df_cuda_gpu_trace_filtered = results["cuda_gpu"]
        df_cuda_kernel_exec_trace = results["cuda_kernel_exec"]
        
        # Take first (sample_size) rows if sample_size is provided
        if sample_size:
            df_nvtx_gpu_proj_trace_filtered = df_nvtx_gpu_proj_trace_filtered.head(sample_size)
            df_cuda_api_trace_filtered = df_cuda_api_trace_filtered.head(sample_size)
            df_cuda_gpu_trace_filtered = df_cuda_gpu_trace_filtered.head(sample_size)
            if (df_cuda_kernel_exec_trace is not None) or (not df_cuda_kernel_exec_trace.empty):
                df_cuda_kernel_exec_trace = df_cuda_kernel_exec_trace.head(sample_size)
        
        df_nvtx_gpu_proj_trace_processed = gpu_and_cpu_times(
                df_nvtx_gpu_proj_trace_filtered,
                df_cuda_api_trace_filtered,
                df_cuda_gpu_trace_filtered,
                df_cuda_kernel_exec_trace,
                category=category,
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e