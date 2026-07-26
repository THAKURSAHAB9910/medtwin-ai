import numpy as np
from typing import Dict, Any, List

from src.agents.state import ClinicalConsensusState, SpecialistOpinion
from src.agents.specialists import (
    RadiologistAgent, CardiologistAgent, NeurologistAgent, PathologistAgent,
    PharmacistAgent, NutritionistAgent, MedicalResearcherAgent, DermatologistAgent
)
from src.rag.medical_kg import MedicalKnowledgeGraph
from src.rag.vector_rag import MedicalRAG

class CoordinatorAgent:
    """
    Coordinator Agent for MedTwin AI.
    Coordinates the multi-agent clinical consensus board.
    Aggregates opinions, detects disagreements, ranks evidence,
    and calls KG and RAG subsystems to build safe, explainable recommendations.
    """
    def __init__(self):
        # Instantiate specialists
        self.specialists = [
            RadiologistAgent(),
            CardiologistAgent(),
            NeurologistAgent(),
            PathologistAgent(),
            PharmacistAgent(),
            NutritionistAgent(),
            MedicalResearcherAgent(),
            DermatologistAgent()
        ]
        # Instantiate KG and RAG
        self.kg = MedicalKnowledgeGraph()
        self.rag = MedicalRAG()

    def run_consensus(self, patient_data: Dict[str, Any]) -> ClinicalConsensusState:
        """
        Executes the consensus workflow over a patient case file.
        """
        state = ClinicalConsensusState(patient_data=patient_data)
        state.history_trace.append("Initialized Multi-Agent State.")
        
        # 1. Specialists independently analyze patient data
        opinions = {}
        for agent in self.specialists:
            opinion = agent.analyze(patient_data)
            opinions[agent.role] = opinion
            state.history_trace.append(f"Agent '{agent.role}' analysis complete.")
            
        state.specialist_opinions = opinions
        
        # 2. Consensus Voting & Disagreement Resolution
        votes: Dict[str, int] = {}
        confidence_sums: Dict[str, float] = {}
        
        for opinion in opinions.values():
            diag = opinion.diagnosis
            votes[diag] = votes.get(diag, 0) + 1
            confidence_sums[diag] = confidence_sums.get(diag, 0.0) + opinion.confidence
            
        # Determine the consensus diagnosis (majority vote)
        # If there's a tie, choose the diagnosis with higher accumulated confidence
        consensus_diagnosis = max(votes, key=lambda d: (votes[d], confidence_sums[d]))
        
        # 3. Identify Disagreements
        disagreements = []
        agree_count = 0
        total_count = len(self.specialists)
        
        for role, opinion in opinions.items():
            if opinion.diagnosis != consensus_diagnosis:
                disagreements.append({
                    "specialist": role,
                    "suggested_diagnosis": opinion.diagnosis,
                    "confidence": opinion.confidence,
                    "evidence": opinion.evidence
                })
            else:
                agree_count += 1
                
        # 4. Calculate Consensus Uncertainty
        # Formula: 1.0 - (fraction of agreeing agents)
        agreement_ratio = agree_count / total_count
        consensus_uncertainty = 1.0 - agreement_ratio
        
        # 5. Aggregate recommended tests and treatments from concurring specialists
        tests_set = set()
        treatments_set = set()
        evidence_list = []
        
        for opinion in opinions.values():
            if opinion.diagnosis == consensus_diagnosis:
                tests_set.update(opinion.recommended_tests)
                treatments_set.update(opinion.treatment_suggestions)
                evidence_list.append(f"{opinion.role}: {opinion.evidence}")
                
        # 6. Query Knowledge Graph for ICD-10 codes and standard recommendations
        icd_codes = self.kg.query_relations(consensus_diagnosis, "has_icd10_code")
        kg_tests = self.kg.query_relations(consensus_diagnosis, "requires_test")
        kg_treatments = self.kg.query_relations(consensus_diagnosis, "has_treatment_class")
        
        # Merge KG insights into outputs
        icd_code = icd_codes[0] if icd_codes else "Unknown"
        tests_set.update(kg_tests)
        
        # 7. Query Vector RAG for guideline citations
        query_text = f"{consensus_diagnosis} " + patient_data.get("clinical_note", "")
        guidelines = self.rag.retrieve_guidelines(query_text, top_k=1)
        rag_citation = guidelines[0]["citation"] if guidelines else "No literature citation found."
        rag_text = guidelines[0]["text"] if guidelines else ""
        
        # Assemble coordinator consensus output
        state.coordinator_consensus = {
            "consensus_diagnosis": consensus_diagnosis,
            "icd10_code": icd_code,
            "consensus_uncertainty": consensus_uncertainty,
            "agreement_rate": agreement_ratio,
            "disagreements": disagreements,
            "evidence_ranking": evidence_list,
            "final_tests": list(tests_set),
            "final_treatments": list(treatments_set),
            "literature_citation": rag_citation,
            "literature_guideline": rag_text
        }
        
        state.history_trace.append("Coordinator consensus aggregation finished.")
        return state

if __name__ == "__main__":
    coordinator = CoordinatorAgent()
    
    # Mock patient file
    patient = {
        "clinical_note": "Patient presents with cough and fever. Lung haziness shown in radiograph.",
        "lab_metrics": {"temperature": 102.5, "wbc_count": 14000.0, "oxygen_saturation": 91.0}
    }
    
    state = coordinator.run_consensus(patient)
    res = state.coordinator_consensus
    print("\nConsensus Engine Verification:")
    print(f" - Consensus Diagnosis: {res['consensus_diagnosis']} ({res['icd10_code']})")
    print(f" - Agreement Rate:      {res['agreement_rate']:.2%}")
    print(f" - Uncertainty:         {res['consensus_uncertainty']:.2%}")
    print(f" - Disagreements:       {len(res['disagreements'])} agents")
    print(f" - RAG Citation:        {res['literature_citation']}")
