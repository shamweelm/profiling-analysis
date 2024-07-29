import pandas as pd
from profiling_analysis.constants import INFERENCE_UNIQUE_KERNEL_NAMES
from profiling_analysis.logger import Logger

log = Logger().get_logger()

# Helper Functions
def filter_data_after_time(df, time_column, ignore_time):
    """Filter the DataFrame to ignore rows before a specified time."""
    try:
        return df[df[time_column] > ignore_time]

    except Exception as e:
        log.error(f"Error: {e}")
        raise e

# Function to map long names to short names
def map_kernel_name(long_name, unique_kernel_names):
    try:
        for short_name in unique_kernel_names:
            if short_name in long_name:
                return short_name
        return long_name

    except Exception as e:
        log.error(f"Error: {e}")
        raise e

def get_unique_kernel_names_inference():
    try:
        # Get unique Kernel Names
        df_unique_kernel_names = pd.read_csv(
            INFERENCE_UNIQUE_KERNEL_NAMES
        )
        unique_kernel_names = df_unique_kernel_names["kernel_name"].tolist()
        
        log.info(f"Unique Kernel Names: {unique_kernel_names}")

        return unique_kernel_names

    except Exception as e:
        log.error(f"Error: {e}")
        raise e

def add_short_kernel_names(df_cuda_kernel_exec_trace, unique_kernel_names):
    try:
        # Get short names
        df_cuda_kernel_exec_trace["Short Kernel Name"] = df_cuda_kernel_exec_trace[
            "Kernel Name"
        ].apply(lambda x: map_kernel_name(x, unique_kernel_names))
        log.info(f"Short Kernel Names added to df_cuda_kernel_exec_trace")
        
        return df_cuda_kernel_exec_trace

    except Exception as e:
        log.error(f"Error: {e}")
        raise e

# Helper function to calculate idle times
def calculate_idle_times(start_times, end_times):
    try:
        idle_times = []
        for i in range(len(start_times) - 1):
            idle_time = start_times[i + 1] - end_times[i]
            if idle_time > 0:
                idle_times.append(idle_time)
        return idle_times

    except Exception as e:
        log.error(f"Error: {e}")
        raise e

# Helper function to get kernels from df_cuda_gpu_trace
def get_kernels_from_gpu_trace(df, start, end):
    try:
        return df[(df['Start (ns)'] >= start) & 
                (df['Start (ns)'] <= end) &
                (df['Kernel Ops'].notnull())]

    except Exception as e:
        log.error(f"Error: {e}")
        raise e