# Constants
import numpy as np

BASE_PATH = "/Users/shamweelmohammed/Desktop/Masters/Dissertation/Trace/"
CATEGORY = "inference_after_model_loading" # inference_after_model_loading or model_loading or all

# inference_llama_repo or 
# inference_llama_repo_torch_ao_int8_wo or 
# inference_llama_repo_custom_int8_ll_wt_no_output or
# inference_llama_repo_custom_int8_ll_wt or 
# inference_torchtune or 
# inference_torchtune_quantize_int4
INFERENCE_TYPE = "inference_llama_repo"
CATEGORY_MAPPING = f"profiling_analysis/configs/category_mapping.json"
INFERENCE_SQLITE_PATH = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/nsys_profile.sqlite"
INFERENCE_CUDA_REPORTS_PATH = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/cuda_stats/"
INFERENCE_NVTX_REPORTS_PATH = BASE_PATH +  f"Inference_Server/{INFERENCE_TYPE}/nvtx_stats/"
INFERENCE_UNIQUE_KERNEL_NAMES = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/unique_kernel_names.csv"

INFERENCE_OPERATIONS_PATH = BASE_PATH + f"Inference_Server/{INFERENCE_TYPE}/operations/"
INFERENCE_OPERATIONS_MAPPING = f"profiling_analysis/configs/{INFERENCE_TYPE}.json"