from profiling_analysis.gpu_cpu_trace import start_gpu_cpu_times_process
from profiling_analysis.logger import Logger
from profiling_analysis.summary import generate_cuda_summaries, generate_nvtx_summaries
from profiling_analysis.trace import get_processed_traces, start_tracing_process
from profiling_analysis.visualize_summaries import visualize_all_summaries_for_inference

log = Logger().get_logger()


def run_inference_analysis(
    trace_process=False,
    gpu_cpu_time_process=False,
    generate_summaries=True,
    sample_size=None,
    visualize=True,
    overwrite=False,
):
    try:
        task = "inference"
        category = None
        if trace_process:
            results = start_tracing_process(task, category, overwrite)
        else:
            results = get_processed_traces(task, category)
        
        if gpu_cpu_time_process:
            start_gpu_cpu_times_process(task, category, results, sample_size)

        
        df_nvtx_gpu_proj_trace_filtered = results["nvtx_gpu_proj"]
        df_nvtx_pushpop_trace_filtered = results["nvtx_pushpop"]
        df_cuda_api_trace_filtered = results["cuda_api"]
        df_cuda_gpu_trace_filtered = results["cuda_gpu"]
        df_cuda_kernel_exec_trace = results["cuda_kernel_exec"]
        

        if generate_summaries:
            generate_cuda_summaries(
                task,
                df_cuda_api_trace_filtered,
                df_cuda_gpu_trace_filtered,
                df_cuda_kernel_exec_trace,
                category,
            )
            generate_nvtx_summaries(
                task,
                df_nvtx_gpu_proj_trace_filtered,
                df_nvtx_pushpop_trace_filtered,
                category,
            )
            
        # Generate Visualizations
        if visualize:
            visualize_all_summaries_for_inference(task, category)

    except Exception as e:
        log.error(f"Error: {e}")
        raise e
