import time
import json
from typing import Dict, Any, List
from src.mcp.server import MCPServer

class MCPClient:
    """
    Model Context Protocol (MCP) Client interface.
    Handles communication with MCP servers, formats payloads, and tracks performance.
    """
    def __init__(self, server: MCPServer = None):
        self.server = server or MCPServer()
        self.call_history: List[Dict[str, Any]] = []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats a JSON-RPC 2.0 request, triggers tool execution, and records telemetry.
        """
        start_time = time.perf_counter()
        
        # Prepare standard JSON-RPC 2.0 payload
        payload = {
            "jsonrpc": "2.0",
            "method": f"tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": len(self.call_history) + 1
        }
        
        response = self.server.call_tool(tool_name, arguments)
        latency = (time.perf_counter() - start_time) * 1000.0  # in ms
        
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": tool_name,
            "arguments": arguments,
            "latency_ms": latency,
            "success": "error" not in response
        }
        self.call_history.append(log_entry)
        
        return response

    def retrieve_full_patient_context(self, patient_id: str) -> Dict[str, Any]:
        """
        Aggregates demographics, lab metrics, PACS findings, and active medications.
        """
        context = {}
        
        # 1. EHR
        ehr_res = self.execute_tool("get_ehr_record", {"patient_id": patient_id})
        if "result" in ehr_res:
            context["ehr"] = ehr_res["result"]
            
        # 2. Labs
        lab_res = self.execute_tool("get_laboratory_results", {"patient_id": patient_id})
        if "result" in lab_res:
            context["labs"] = lab_res["result"]
            
        # 3. PACS
        pacs_res = self.execute_tool("get_radiology_pacs", {"patient_id": patient_id})
        if "result" in pacs_res:
            context["pacs"] = pacs_res["result"]
            
        # 4. Pharmacy
        pharm_res = self.execute_tool("get_pharmacy_records", {"patient_id": patient_id})
        if "result" in pharm_res:
            context["pharmacy"] = pharm_res["result"]
            
        # 5. Cloud notes/archives
        cloud_res = self.execute_tool("get_cloud_archives", {"patient_id": patient_id})
        if "result" in cloud_res:
            context["cloud"] = cloud_res["result"]
            
        # 6. Hospital management
        hosp_res = self.execute_tool("get_hospital_management", {"patient_id": patient_id})
        if "result" in hosp_res:
            context["hospital"] = hosp_res["result"]

        return context
