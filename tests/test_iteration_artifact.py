"""
Tests for the iteration_artifact module.
"""
import unittest
from unittest.mock import Mock
from llm_opt.core.iteration_artifact import IterationArtifact
from llm_opt.core.performance_report import PerformanceReport


class TestIterationArtifact(unittest.TestCase):
    """Test the IterationArtifact class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock performance report
        self.mock_perf_report = Mock(spec=PerformanceReport)
        self.mock_perf_report.desc.return_value = "Mock performance report"
        self.mock_perf_report.speedup_medians.return_value = 3.5
        
        # Create sample C code
        self.c_code = """
        void test_func(double* a, int a_size, double* b, int b_size, double* output, int output_size) {
            for (int i = 0; i < a_size && i < output_size; i++) {
                output[i] = a[i] + b[i];
            }
        }
        """
        
        # Create sample prompt and response
        self.prompt = "Generate a C function to add two arrays"
        self.response = "Here's a C function to add two arrays:\n```c\n" + self.c_code + "\n```"
        
        # Create an artifact
        self.artifact = IterationArtifact(
            idx=1,
            c_code=self.c_code,
            prompt=self.prompt,
            response=self.response,
            performance_report=self.mock_perf_report,
            success=True,
            error=None
        )

    def test_init(self):
        """Test artifact initialization."""
        artifact = IterationArtifact(
            idx=2,
            c_code=self.c_code,
            prompt=self.prompt,
            response=self.response
        )
        
        self.assertEqual(artifact.idx, 2)
        self.assertEqual(artifact.c_code, self.c_code)
        self.assertEqual(artifact.prompt, self.prompt)
        self.assertEqual(artifact.response, self.response)
        self.assertIsNone(artifact.performance_report)
        self.assertIsNone(artifact.success)
        self.assertIsNone(artifact.error)

    def test_init_with_all_params(self):
        """Test artifact initialization with all parameters."""
        artifact = IterationArtifact(
            idx=3,
            c_code=self.c_code,
            prompt=self.prompt,
            response=self.response,
            performance_report=self.mock_perf_report,
            success=False,
            error="Some error"
        )
        
        self.assertEqual(artifact.idx, 3)
        self.assertEqual(artifact.c_code, self.c_code)
        self.assertEqual(artifact.prompt, self.prompt)
        self.assertEqual(artifact.response, self.response)
        self.assertEqual(artifact.performance_report, self.mock_perf_report)
        self.assertEqual(artifact.success, False)
        self.assertEqual(artifact.error, "Some error")

    def test_short_desc_success(self):
        """Test short description of a successful artifact."""
        desc = self.artifact.short_desc()
        
        # Check that all required parts are in the description
        self.assertIn("Iteration 1", desc)
        self.assertIn("Implementation:", desc)
        self.assertIn("Success: True", desc)
        self.assertIn("Error: None", desc)
        self.assertIn("Performance Report: Mock performance report", desc)
        
        # Check that the C code is included
        self.assertIn("output[i] = a[i] + b[i]", desc)

    def test_short_desc_failure(self):
        """Test short description of a failed artifact."""
        artifact = IterationArtifact(
            idx=4,
            c_code=self.c_code,
            prompt=self.prompt,
            response=self.response,
            success=False,
            error="Compilation error"
        )
        
        desc = artifact.short_desc()
        
        self.assertIn("Iteration 4", desc)
        self.assertIn("Success: False", desc)
        self.assertIn("Error: Compilation error", desc)
        self.assertIn("Performance Report: N/A", desc)

    def test_short_desc_no_performance_report(self):
        """Test short description without a performance report."""
        artifact = IterationArtifact(
            idx=5,
            c_code=self.c_code,
            prompt=self.prompt,
            response=self.response,
            success=True
        )
        
        desc = artifact.short_desc()
        
        self.assertIn("Performance Report: N/A", desc)


if __name__ == "__main__":
    unittest.main()