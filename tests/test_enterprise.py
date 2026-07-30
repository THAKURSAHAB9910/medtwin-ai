import pytest
import json
from src.mcp.server import MCPServer
from src.mcp.client import MCPClient
from src.agents.langgraph_workflow import ClinicalWorkflowGraph, AgentState
from src.automation.n8n_integration import N8NAutomationEngine
from src.database.schemas import RedisCacheSimulation, FHIRPatient, FHIRObservation

def test_mcp_server_and_client():
    """Verify MCP server tools listing and client JSON-RPC executions."""
    server = MCPServer()
    client = MCPClient(server)
    
    # 1. Test Tools list schema
    tools = server.list_tools()
    assert len(tools) > 0
    assert any(t["name"] == "get_ehr_record" for t in tools)
    
    # 2. Test tool execution (success path)
    ehr_res = client.execute_tool("get_ehr_record", {"patient_id": "PT-8891"})
    assert "result" in ehr_res
    assert ehr_res["result"]["patient_id"] == "PT-8891"
    assert "fhir_patient" in ehr_res["result"]
    
    # 3. Test tool execution (error path)
    bad_res = client.execute_tool("get_ehr_record", {"patient_id": "BAD-ID"})
    assert "error" in bad_res
    assert bad_res["error"]["code"] == -32602

def test_patient_context_aggregation():
    """Verify client retrieves full structured patient context."""
    server = MCPServer()
    client = MCPClient(server)
    
    context = client.retrieve_full_patient_context("PT-8891")
    assert "ehr" in context
    assert "labs" in context
    assert "pacs" in context
    assert "pharmacy" in context
    assert context["labs"]["vitals"]["temperature"] == 101.5
    assert context["labs"]["vitals"]["oxygen_saturation"] == 91.0

def test_langgraph_workflow_high_risk_pause():
    """Verify LangGraph workflow graph executes and pauses on high-risk hypoxia review."""
    server = MCPServer()
    client = MCPClient(server)
    graph = ClinicalWorkflowGraph(client)
    
    # PT-8891 is high risk (oxygen saturation = 91% < 92%)
    state = AgentState(patient_id="PT-8891", query="Query patient record.")
    final_state = graph.execute_workflow(state)
    
    # Review status should pause for human approval
    assert final_state.review_status == "pending_human_approval"
    assert not final_state.human_approved
    assert final_state.diagnosis == "Community-Acquired Pneumonia"
    assert final_state.risk_assessment["high_risk"]
    
    # Resuming workflow after human approval override
    final_state.human_approved = True
    final_state.review_status = "approved"
    resumed_state = graph.execute_workflow(final_state)
    
    assert resumed_state.review_status == "completed"
    assert "patient_id" in resumed_state.final_report
    assert resumed_state.final_report["diagnosis"] == "Community-Acquired Pneumonia"

def test_n8n_automations_escalations():
    """Verify n8n post-diagnostic actions list and high-risk dispatch alerts."""
    engine = N8NAutomationEngine()
    
    mock_report = {
        "patient_id": "PT-8891",
        "diagnosis": "Pneumonia",
        "treatment_plan": ["Oxygen therapy"]
    }
    
    # 1. Normal run (no high-risk)
    log = engine.trigger_post_diagnosis_workflow(mock_report, is_high_risk=False)
    assert len(log["escalations"]) == 0
    assert any(s["step"] == "Notify Pharmacy (Order Prescriptions)" for s in log["steps_executed"])
    
    # 2. High risk run (should trigger emergency team alerts)
    high_log = engine.trigger_post_diagnosis_workflow(mock_report, is_high_risk=True)
    assert len(high_log["escalations"]) > 0
    assert any("Emergency Medical Team" in s["step"] for s in high_log["steps_executed"])

def test_redis_cache_and_pydantic_schemas():
    """Verify Redis simulation cache set, get, TTL, and FHIR validation."""
    cache = RedisCacheSimulation()
    assert cache.ping()
    
    cache.set("test_key", "test_val", ex=2)
    assert cache.get("test_key") == "test_val"
    assert cache.exists("test_key")
    
    cache.delete("test_key")
    assert cache.get("test_key") is None
    
    # Test FHIR Patient schema validation
    pat = FHIRPatient(
        id="PT-8891",
        name=[{"family": "Doe", "given": ["Jane"]}],
        gender="female",
        birthDate="1978-05-14"
    )
    assert pat.resourceType == "Patient"
