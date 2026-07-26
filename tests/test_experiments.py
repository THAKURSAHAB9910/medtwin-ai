import pytest
import os

from src.experiments.framework import ResearchExperimentFramework
from src.experiments.reporter import ExperimentReporter

def test_experiments_framework():
    framework = ResearchExperimentFramework()
    results = framework.run_all()
    
    assert len(results) == 4
    assert "multimodal_reasoning" in results
    assert "lora_efficiency" in results
    assert "clinical_rag_hallucination" in results
    assert "synthetic_robustness" in results
    
    for name, details in results.items():
        assert "research_question" in details
        assert "hypothesis" in details
        assert "baseline" in details
        assert "improved_model" in details
        assert "dataset" in details
        assert "metrics" in details
        assert "results" in details
        assert "discussion" in details
        assert "conclusion" in details

def test_experiment_reporter(tmp_path):
    framework = ResearchExperimentFramework()
    results = framework.run_all()
    
    reporter = ExperimentReporter(results)
    
    # Test markdown report compilation
    output_md = tmp_path / "test_report.md"
    content = reporter.generate_markdown_report(str(output_md))
    
    assert os.path.exists(output_md)
    assert "# MedTwin AI: Clinical Research Experiments Report" in content
    assert "Multimodal Diagnostics Evaluation" in content
    
    # Test comparative charts export
    output_png = tmp_path / "test_results.png"
    reporter.generate_comparison_charts(str(output_png))
    assert os.path.exists(output_png)
