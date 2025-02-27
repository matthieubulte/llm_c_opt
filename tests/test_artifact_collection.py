"""
Tests for the artifact_collection module.
"""
import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

from llm_opt.utils.artifact_collection import ArtifactCollection
from llm_opt.core.iteration_artifact import IterationArtifact
from llm_opt.core.performance_report import PerformanceReport


class TestArtifactCollection(unittest.TestCase):
    """Test the ArtifactCollection class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.func_name = "test_func"
        
        # Create a collection
        self.collection = ArtifactCollection(self.temp_dir, self.func_name)
        
        # Create mock performance reports with different speedups
        self.perf_report1 = Mock(spec=PerformanceReport)
        self.perf_report1.desc.return_value = "Performance report 1"
        self.perf_report1.speedup_medians.return_value = 2.0
        
        self.perf_report2 = Mock(spec=PerformanceReport)
        self.perf_report2.desc.return_value = "Performance report 2"
        self.perf_report2.speedup_medians.return_value = 3.0  # Faster than perf_report1
        
        # Create some sample artifacts
        self.artifact1 = IterationArtifact(
            idx=1,
            c_code="void test_func() { /* implementation 1 */ }",
            prompt="Generate implementation 1",
            response="Here's implementation 1",
            performance_report=self.perf_report1,
            success=True
        )
        
        self.artifact2 = IterationArtifact(
            idx=2,
            c_code="void test_func() { /* implementation 2 */ }",
            prompt="Generate implementation 2",
            response="Here's implementation 2",
            performance_report=self.perf_report2,
            success=True
        )
        
        self.failed_artifact = IterationArtifact(
            idx=3,
            c_code="void test_func() { /* failed implementation */ }",
            prompt="Generate failed implementation",
            response="Here's a failed implementation",
            success=False,
            error="Compilation error"
        )

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test collection initialization."""
        collection = ArtifactCollection(self.temp_dir, self.func_name)
        
        self.assertEqual(collection.run_dir, self.temp_dir)
        self.assertEqual(collection.func_name, self.func_name)
        self.assertEqual(len(collection.artifacts), 0)

    def test_add_artifact(self):
        """Test adding an artifact to the collection."""
        self.collection.add_artifact(self.artifact1)
        
        self.assertEqual(len(self.collection.artifacts), 1)
        self.assertEqual(self.collection.artifacts[0], self.artifact1)
        
        # Check that the files were created
        iteration_dir = os.path.join(self.temp_dir, "iterations", "iteration_1")
        self.assertTrue(os.path.exists(iteration_dir))
        self.assertTrue(os.path.exists(os.path.join(iteration_dir, f"{self.func_name}.c")))
        self.assertTrue(os.path.exists(os.path.join(iteration_dir, "prompt.txt")))
        self.assertTrue(os.path.exists(os.path.join(iteration_dir, "response.txt")))
        self.assertTrue(os.path.exists(os.path.join(iteration_dir, "artifact.txt")))

    def test_to_str(self):
        """Test converting the collection to a string."""
        self.collection.add_artifact(self.artifact1)
        self.collection.add_artifact(self.artifact2)
        
        result = self.collection.to_str()
        
        # Check that the string contains both artifacts
        self.assertIn("implementation 1", result)
        self.assertIn("implementation 2", result)
        self.assertIn("Iteration 1", result)
        self.assertIn("Iteration 2", result)

    def test_get_best_artifact(self):
        """Test getting the best artifact."""
        # Add artifacts in reverse order of performance
        self.collection.add_artifact(self.artifact1)  # Speedup: 2.0
        self.collection.add_artifact(self.failed_artifact)  # Failed, no speedup
        self.collection.add_artifact(self.artifact2)  # Speedup: 3.0
        
        best = self.collection.get_best_artifact()
        
        # artifact2 should be the best (highest speedup)
        self.assertEqual(best, self.artifact2)

    def test_get_best_artifact_with_failures(self):
        """Test getting the best artifact when some artifacts failed."""
        # Only add failed artifacts
        self.collection.add_artifact(self.failed_artifact)
        
        best = self.collection.get_best_artifact()
        
        # The failed artifact should be returned, even though it has no speedup
        self.assertEqual(best, self.failed_artifact)

    @patch("llm_opt.utils.artifact_collection.to_html")
    def test_checkpoint(self, mock_to_html):
        """Test checkpointing artifacts."""
        self.collection.add_artifact(self.artifact1)
        
        # Set success to None to test automatic setting of success
        self.collection.artifacts[-1].success = None
        
        self.collection.checkpoint()
        
        # Success should be set to True automatically
        self.assertTrue(self.collection.artifacts[-1].success)
        
        # Check that to_html was called
        mock_to_html.assert_called_once()
        
        # Check that the best artifact was saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "best_artifact.txt")))


if __name__ == "__main__":
    unittest.main()