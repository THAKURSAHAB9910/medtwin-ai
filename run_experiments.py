from src.experiments.framework import ResearchExperimentFramework
from src.experiments.reporter import ExperimentReporter

def run_research_suite():
    print("="*75)
    print(" MEDTWIN AI: MULTI-TASK CLINICAL RESEARCH EXPERIMENTS SUITE ")
    print("="*75)
    
    # 1. Initialize orchestrator
    print("[1/3] Initializing research experiments framework...")
    framework = ResearchExperimentFramework()
    
    # 2. Run experiments
    print("[2/3] Executing 4 target research validation studies...")
    results = framework.run_all()
    
    for exp_name, details in results.items():
        print(f"\n --- EXPERIMENT: {exp_name.upper().replace('_', ' ')} ---")
        print(f"  * Question:   {details['research_question']}")
        print(f"  * Hypothesis: {details['hypothesis']}")
        print(f"  * Baseline:   {details['baseline']}")
        print(f"  * Improved:   {details['improved_model']}")
        print(f"  * Conclusion: {details['conclusion']}")
        
    # 3. Compile reports
    print("\n[3/3] Generating scientific documents & plots...")
    reporter = ExperimentReporter(results)
    reporter.generate_markdown_report("docs/experimental_report.md")
    reporter.generate_comparison_charts("docs/experimental_results.png")
    
    print("\n" + "="*75)
    print(" Research Experiments Suite Execution Complete successfully. ")
    print("="*75)

if __name__ == "__main__":
    run_research_suite()
