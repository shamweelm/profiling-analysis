from profiling_analysis.logger import Logger
from profiling_analysis.summary import generate_cuda_summaries, generate_nvtx_summaries
from profiling_analysis.trace import trace_after_model_loading_inference

log = Logger().get_logger()


def run_inference_analysis():
    try:
        task = "inference"
        results = trace_after_model_loading_inference()

        df_nvtx_gpu_proj_trace_filtered = results["nvtx_gpu_proj"]
        df_nvtx_pushpop_trace_filtered = results["nvtx_pushpop"]
        df_cuda_api_trace_filtered = results["cuda_api"]
        df_cuda_gpu_trace_filtered = results["cuda_gpu"]
        df_cuda_kernel_exec_trace = results["cuda_kernel_exec"]
        
        generate_cuda_summaries(task, df_cuda_api_trace_filtered, df_cuda_gpu_trace_filtered, df_cuda_kernel_exec_trace)
        generate_nvtx_summaries(task, df_nvtx_gpu_proj_trace_filtered, df_nvtx_pushpop_trace_filtered)
        
    except Exception as e:
        log.error(f"Error: {e}")
        raise e