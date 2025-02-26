from typing import Any, Optional
from llm_opt.core.performance_report import PerformanceReport


class IterationArtifact:
    def __init__(
        self,
        idx: int,
        c_code: str,
        prompt: str,
        response: str,
        performance_report: Optional[PerformanceReport] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.idx = idx
        self.c_code = c_code
        self.prompt = prompt
        self.response = response
        self.performance_report = performance_report
        self.success = success
        self.error = error

    def short_desc(self):
        return (
            ("=" * 80)
            + f"""
Iteration {self.idx}
Implementation: {self.c_code}
Success: {str(self.success)}
Error: {self.error}
Performance Report: {self.performance_report.desc() if self.performance_report else "N/A"}
        """
        )
