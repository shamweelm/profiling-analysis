import os
import re
import json
import pandas as pd
from profiling_analysis.configs.constants import END_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING, END_TIME_FOR_INFERENCE_ALL, END_TIME_FOR_INFERENCE_BEFORE_MODEL_LOADING, INFERENCE_SQLITE_PATH, INFERENCE_UNIQUE_KERNEL_NAMES, START_TIME_FOPR_INFERENCE_BEFORE_MODEL_LOADING, START_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING, START_TIME_FOR_INFERENCE_ALL
from datetime import datetime
from profiling_analysis import logger
import sqlite3
import pandas as pd

# Helper Functions
def filter_data_between_time(df, time_column, start_time, end_time):
    """Filter the DataFrame to ignore rows before a specified time."""
    try:
        # return df[df[time_column] > ignore_time]
        return df[(df[time_column] >= start_time) & (df[time_column] <= end_time)]

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

# Function to map long names to short names
def map_kernel_name(long_name, unique_kernel_names):
    try:
        for short_name in unique_kernel_names:
            if short_name in long_name:
                return short_name
        return long_name

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    

def extract_unique_kernel_names(sqlite_db_path, unique_kernel_names_path):
    try:
        if os.path.exists(unique_kernel_names_path):
            logger.info(f"Unique Kernel Names already extracted at {unique_kernel_names_path}")
            return unique_kernel_names_path
        
        if not os.path.exists(sqlite_db_path):
            raise FileNotFoundError(f"SQLite database not found at {sqlite_db_path}")
        
        # Establishing a connection to the SQLite database
        conn = sqlite3.connect(sqlite_db_path)  # Replace with your database path

        # Defining the query
        query = """
        SELECT s.value AS kernel_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL c
        JOIN StringIds s ON c.shortName = s.id;
        """

        # Executing the query and reading the results into a pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Closing the connection
        conn.close()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Save the unique kernel names to a CSV file
        df.to_csv(unique_kernel_names_path, index=False)
        logger.info(f"Unique Kernel Names saved to {unique_kernel_names_path}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def get_unique_kernel_names_inference():
    try:
        extract_unique_kernel_names(INFERENCE_SQLITE_PATH, INFERENCE_UNIQUE_KERNEL_NAMES)
        
        # Get unique Kernel Names
        df_unique_kernel_names = pd.read_csv(
            INFERENCE_UNIQUE_KERNEL_NAMES
        )
        unique_kernel_names = df_unique_kernel_names["kernel_name"].tolist()
        
        # Remove duplicates
        unique_kernel_names = list(set(unique_kernel_names))
        
        logger.info(f"Unique Kernel Names: {unique_kernel_names}")

        return unique_kernel_names

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

def add_short_kernel_names(df_cuda_kernel_exec_trace, unique_kernel_names):
    try:
        # Get short names
        df_cuda_kernel_exec_trace["Short Kernel Name"] = df_cuda_kernel_exec_trace[
            "Kernel Name"
        ].apply(lambda x: map_kernel_name(x, unique_kernel_names))
        logger.info(f"Short Kernel Names added to df_cuda_kernel_exec_trace")
        
        return df_cuda_kernel_exec_trace

    except Exception as e:
        logger.error(f"Error: {e}")
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
        logger.error(f"Error: {e}")
        raise e

# Helper function to get kernels from df_cuda_gpu_trace
def get_kernels_from_gpu_trace(df, start, end):
    try:
        return df[(df['Start (ns)'] >= start) & 
                (df['Start (ns)'] <= end) &
                (df['Kernel Ops'].notnull())]

    except Exception as e:
        logger.error(f"Error: {e}")
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
        logger.error(f"Error: {e}")
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
        logger.error(f"Error: {e}")
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
        logger.error(f"Error: {e}")
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
        logger.error(f"Error: {e}")
        raise e
    

def convert_string_to_dataframe(json_string):
    try:
        # Step 1: Replace single quotes with double quotes
        json_string = json_string.replace("'", '"')
        
        # Step 2: Fix for `nan` values, replacing them with null (JSON equivalent of NaN)
        json_string = re.sub(r'\bnan\b', 'null', json_string)
        
        # Replace None with null
        json_string = re.sub(r'\bNone\b', 'null', json_string)
        
        # Step 3: Parse the cleaned JSON string into a Python object
        list_of_dicts = json.loads(json_string)
        
        # Step 4: Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(list_of_dicts)

        return df
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing JSON string: {e}")
        return None
    

# {
#     "operation_token_embedding" : "tok_embeddings: ParallelEmbedding"
# }
def get_operation_mapping(operation_mapping_path):
    try:
        with open(operation_mapping_path) as f:
            operation_mapping = json.load(f)
        
        return operation_mapping
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    

def get_start_and_end_times(category):
    try:
        if category == "inference_after_model_loading":
            start_time = START_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING
            end_time = END_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING
        elif category == "inference_before_model_loading":
            start_time = START_TIME_FOPR_INFERENCE_BEFORE_MODEL_LOADING
            end_time = END_TIME_FOR_INFERENCE_BEFORE_MODEL_LOADING
        else:
            start_time = START_TIME_FOR_INFERENCE_ALL
            end_time = END_TIME_FOR_INFERENCE_ALL
        
        return start_time, end_time
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e