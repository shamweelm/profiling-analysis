from profiling_analysis.analysis import run_inference_analysis

if __name__ == "__main__":
    # run_inference_analysis(
    #     trace_process=False,
    #     gpu_cpu_time_process=False,
    #     generate_summaries=False,
    #     sample_size=10000,
    #     visualize=True,
    #     overwrite=True,
    # )
    run_inference_analysis(
        trace_process=True,
        gpu_cpu_time_process=True,
        generate_summaries=True,
        sample_size=10000,
        visualize=True,
        overwrite=True,
    )