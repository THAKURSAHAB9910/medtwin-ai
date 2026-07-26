import os
import numpy as np
import matplotlib.pyplot as plt

from src.twin.simulation import PatientDigitalTwinEngine
from src.evaluation.twin_evaluator import PatientTwinEvaluator

def generate_twin_visualization(
    sim_res: dict, 
    historical: dict, 
    output_path: str = "docs/patient_twin_longitudinal_forecast.png"
):
    """
    Plots longitudinal forecasted trajectories with 95% shaded credible intervals,
    comparing them side-by-side with historical patient records.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    months = sim_res["trajectories"]["months"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vitals_to_plot = [
        ("weight", "Weight (kg)", "olive"),
        ("blood_glucose", "Blood Glucose (mg/dL)", "crimson"),
        ("blood_pressure", "Blood Pressure (mmHg)", "navy")
    ]
    
    for idx, (vital, label, color) in enumerate(vitals_to_plot):
        mean = np.array(sim_res["trajectories"][vital]["mean"])
        std = np.array(sim_res["trajectories"][vital]["std"])
        actual = np.array(historical[vital])
        
        # 95% Credible Interval bounds
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        
        # Plot simulated mean and shaded interval
        axes[idx].plot(months, mean, color=color, label="Simulated Twin (Mean)", linewidth=2)
        axes[idx].fill_between(months, lower, upper, color=color, alpha=0.15, label="95% Credible Interval")
        
        # Plot historical actuals
        axes[idx].plot(months, actual, "ko--", label="Historical Actuals", markersize=5, alpha=0.8)
        
        axes[idx].set_xlabel("Time (Months)")
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f"Patient {label} Trajectory")
        axes[idx].legend()
        axes[idx].grid(True, linestyle="--")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Longitudinal digital twin forecast plot saved to: {output_path}")


def run_twin_pipeline():
    print("="*75)
    print(" MEDTWIN AI: PATIENT DIGITAL TWIN SIMULATION LIFECYCLE ")
    print("="*75)
    
    # 1. Baseline patient vitals (obese diabetic profile)
    print("[1/4] Configuring patient vitals baseline...")
    vitals = {
        "weight": 95.0,
        "blood_pressure": 142.0,
        "blood_glucose": 170.0,
        "heart_rate": 82.0,
        "wbc_count": 7800.0,
        "oxygen_saturation": 96.0
    }
    
    # 2. What-if intervention scenario over 12 months
    # Patient targets: Lose 10kg, start Metformin, exercise 4h/week, restrict sodium
    print("[2/4] Defining What-If intervention scenario (Weight loss + Metformin)...")
    scenario = {
        "weight_change": -10.0,
        "metformin_mg": 1000.0,
        "beta_blocker_mg": 0.0,
        "exercise_hours": 4.0,
        "sodium_intake_g": 1.5
    }
    
    # 3. Execute Simulation Engine
    print("\n[3/4] Running Bayesian state-space trajectory projection...")
    engine = PatientDigitalTwinEngine()
    sim_res = engine.run_what_if_simulation(vitals, scenario, steps=12)
    
    # Generate mock historical patient actuals showing similar path but with stochastic deviations
    # representing real physiological tracking data
    np.random.seed(101)
    steps = 12
    historical = {
        "weight": [95.0 - (10.0/steps)*t + np.random.normal(0, 0.4) for t in range(steps + 1)],
        "blood_pressure": [142.0 - 0.9 * ((10.0/steps)*t) + np.random.normal(0, 1.2) for t in range(steps + 1)],
        "blood_glucose": [170.0 - 1.5 * t * (1000.0/500.0) + np.random.normal(0, 2.5) for t in range(steps + 1)]
    }
    
    # Force first step match
    historical["weight"][0] = vitals["weight"]
    historical["blood_pressure"][0] = vitals["blood_pressure"]
    historical["blood_glucose"][0] = vitals["blood_glucose"]
    
    # 4. Evaluate and Benchmark
    print("\n[4/4] Performing longitudinal accuracy and calibration evaluations...")
    evaluator = PatientTwinEvaluator()
    eval_metrics = evaluator.evaluate_trajectories(sim_res["trajectories"], historical)
    evaluator.print_twin_report(eval_metrics)
    
    # Print clinical forecast metrics
    print("\n" + "="*70)
    print(" DIGITAL TWIN PROJECTIONS & RISK ANALYSIS ")
    print("="*70)
    print(f"- Cardiovascular Risk:            {sim_res['risk_scores']['cardiovascular_risk']:.2%}")
    print(f"- Readmission Probability:        {sim_res['risk_scores']['hospital_readmission_risk']:.2%}")
    print(f"- Expected Recovery Time:         {sim_res['risk_scores']['predicted_recovery_time_days']:.1f} days")
    print(f"- Possible Complications:         {sim_res['complications']}")
    print("-"*70)
    print("Literature Guidelines Citations:")
    for cit in sim_res["guideline_citations"]:
        print(f" * {cit}")
    print("="*70)
    
    # Generate visualization
    generate_twin_visualization(sim_res, historical)
    print("\nPatient Digital Twin Simulation Lifecycle Complete.")

if __name__ == "__main__":
    run_twin_pipeline()
