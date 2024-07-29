import os
import pandas as pd
from profiling_analysis.constants import INFERENCE_UNIQUE_KERNEL_NAMES
from profiling_analysis.logger import Logger
from datetime import datetime

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


def get_trace_path(base_path, trace_name, category=None):
    try:
        trace_base_path = base_path
        
        if not os.path.exists(trace_base_path):
            os.makedirs(trace_base_path, exist_ok=True)
        
        if category:
            if not os.path.exists(os.path.join(trace_base_path, category)):
                os.makedirs(os.path.join(trace_base_path, category), exist_ok=True)
            
            trace_path = os.path.join(trace_base_path, category, trace_name)
        else:
            trace_path = os.path.join(trace_base_path, trace_name)
        
        return trace_path
    
    except Exception as e:
        log.error(f"Error: {e}")
        raise e


def get_reports_path(base_path, report_name, category=None):
    try:
        # reports_base_path = os.path.join(base_path, "results", f"reports_{datetime.now().strftime('%Y_%m_%d')}")
        reports_base_path = os.path.join(base_path, "results", f"{datetime.now().strftime('%Y_%m_%d')}", "reports")
        
        if not os.path.exists(reports_base_path):
            os.makedirs(reports_base_path, exist_ok=True)
        
        # reports_path = os.path.join(reports_base_path, report_name)
        if category:
            if not os.path.exists(os.path.join(reports_base_path, category)):
                os.makedirs(os.path.join(reports_base_path, category), exist_ok=True)
            
            reports_path = os.path.join(reports_base_path, category, report_name)
        else:
            reports_path = os.path.join(reports_base_path, report_name)
        
        return reports_path
    
    except Exception as e:
        log.error(f"Error: {e}")
        raise e
    

def get_summary_path(base_path, summary_name, category=None):
    try:
        # summary_base_path = os.path.join(base_path, "results", f"summaries_{datetime.now().strftime('%Y_%m_%d')}")
        summary_base_path = os.path.join(base_path, "results", f"{datetime.now().strftime('%Y_%m_%d')}", "summaries")
        
        if not os.path.exists(summary_base_path):
            os.makedirs(summary_base_path, exist_ok=True)
        
        # summary_path = os.path.join(summary_base_path, summary_name)
        if category:
            if not os.path.exists(os.path.join(summary_base_path, category)):
                os.makedirs(os.path.join(summary_base_path, category), exist_ok=True)
            
            summary_path = os.path.join(summary_base_path, category, summary_name)
        else:
            summary_path = os.path.join(summary_base_path, summary_name)
        
        return summary_path
    
    except Exception as e:
        log.error(f"Error: {e}")
        raise e


def get_visualization_path(base_path, filename, operation, category=None):
    try:
        # visualization_base_path = os.path.join(base_path, "results", f"visualizations_{datetime.now().strftime('%Y_%m_%d')}")
        visualization_base_path = os.path.join(base_path, "results", f"{datetime.now().strftime('%Y_%m_%d')}", "visualizations")
        
        if not os.path.exists(visualization_base_path):
            os.makedirs(visualization_base_path, exist_ok=True)
        
        # visualization_path = os.path.join(visualization_base_path, filename)
        if category:
            if not os.path.exists(os.path.join(visualization_base_path, category)):
                os.makedirs(os.path.join(visualization_base_path, category), exist_ok=True)
            
            if not os.path.exists(os.path.join(visualization_base_path, category, operation)):
                os.makedirs(os.path.join(visualization_base_path, category, operation), exist_ok=True)
                
            visualization_path = os.path.join(visualization_base_path, category, operation, filename)
        else:
            if not os.path.exists(os.path.join(visualization_base_path, operation)):
                os.makedirs(os.path.join(visualization_base_path, operation), exist_ok=True)
            
            visualization_path = os.path.join(visualization_base_path, operation, filename)
        
        return visualization_path
    
    except Exception as e:
        log.error(f"Error: {e}")
        raise e