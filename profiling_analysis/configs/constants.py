# Constants
import numpy as np

BASE_PATH = "/Users/shamweelmohammed/Desktop/Masters/Dissertation/Trace/"
CATEGORY = "inference_before_model_loading" # inference_after_model_loading or inference_before_model_loading or all
START_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING = 15.8e9 # 13.7 seconds # 15.8
END_TIME_FOR_INFERENCE_AFTER_MODEL_LOADING = np.inf
START_TIME_FOPR_INFERENCE_BEFORE_MODEL_LOADING = 0
END_TIME_FOR_INFERENCE_BEFORE_MODEL_LOADING = 15.8e9
START_TIME_FOR_INFERENCE_ALL = 0
END_TIME_FOR_INFERENCE_ALL = np.inf

INFERENCE_TYPE = "inference_llama_repo" # inference_torchtune or inference_llama_repo
INFERENCE_SQLITE_PATH = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/nsys_profile.sqlite"
INFERENCE_CUDA_REPORTS_PATH = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/cuda_stats/"
INFERENCE_NVTX_REPORTS_PATH = BASE_PATH +  f"Inference_Server/{INFERENCE_TYPE}/nvtx_stats/"
INFERENCE_UNIQUE_KERNEL_NAMES = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/unique_kernel_names.csv"

INFERENCE_OPERATIONS_PATH = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/operations/"
INFERENCE_OPERATIONS_MAPPING = "profiling_analysis/configs/inference_llama.json"