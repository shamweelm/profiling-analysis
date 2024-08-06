from profiling_analysis.configs.constants import CATEGORY
from profiling_analysis.gpu_cpu_trace import start_gpu_cpu_times_process
from profiling_analysis.summary import generate_cuda_summaries, generate_nvtx_summaries
from profiling_analysis.trace import get_processed_traces, start_tracing_process
from profiling_analysis.visualize_operations import visualize_all_operations
from profiling_analysis.visualize_summaries import visualize_all_summaries_for_inference
from profiling_analysis import logger

def run_inference_analysis(
    trace_process=False,
    gpu_cpu_time_process=False,
    generate_summaries=True,
    sample_size=None,
    visualize=True,
    overwrite=False,
):
    try:
        logger.info("Running Inference Analysis")
        task = "inference"
        category = CATEGORY
        
        if category == "all" and sample_size is not None:
            sample_size = sample_size * 5
        
        results = None
        if trace_process:
            logger.info("Processing Trace")
            results = start_tracing_process(task, category, overwrite)
        elif trace_process == False and (gpu_cpu_time_process == True or generate_summaries == True):
            logger.info("Skipping Trace Processing and using existing traces")
            results = get_processed_traces(task, category)
        
        if gpu_cpu_time_process:
            logger.info("Processing GPU and CPU Times")
            start_gpu_cpu_times_process(task, category, results, sample_size)

        
        if results:
            df_nvtx_gpu_proj_trace_filtered = results["nvtx_gpu_proj"]
            df_nvtx_pushpop_trace_filtered = results["nvtx_pushpop"]
            df_cuda_api_trace_filtered = results["cuda_api"]
            df_cuda_gpu_trace_filtered = results["cuda_gpu"]
            df_cuda_kernel_exec_trace = results["cuda_kernel_exec"]
        

        if generate_summaries:
            logger.info("Generating Summaries")
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
            logger.info("Visualizing")
            visualize_all_summaries_for_inference(task, category)
            visualize_all_operations(task, category)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
