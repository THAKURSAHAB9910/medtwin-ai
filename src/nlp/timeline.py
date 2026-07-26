from datetime import datetime
from typing import List, Dict, Any

class ClinicalTimelineCompiler:
    """
    Parses scattered clinical events (Prescriptions, Radiology Reports, Lab Vitals)
    and compiles a chronological longitudinal patient history.
    """
    def compile_timeline(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts medical reports chronologically based on their recorded date stamps.
        """
        def parse_date(date_str: str) -> datetime:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return datetime.min

        # Sort in-place
        sorted_events = sorted(events, key=lambda x: parse_date(x.get("date", "")))
        return sorted_events

    def generate_narrative_timeline(self, events: List[Dict[str, Any]]) -> str:
        """
        Generates a readable clinical text summary tracing patient history.
        """
        sorted_events = self.compile_timeline(events)
        
        narratives = []
        for ev in sorted_events:
            date = ev.get("date", "Unknown Date")
            ev_type = ev.get("type", "General Note").upper()
            content = ev.get("content", "")
            
            # Simple summarization mapper
            summary = content[:80] + "..." if len(content) > 80 else content
            narratives.append(f"[{date}] - {ev_type}: {summary}")
            
        return "\n".join(narratives)
