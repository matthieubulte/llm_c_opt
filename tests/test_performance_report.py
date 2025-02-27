"""
Tests for the performance_report module.
"""
import unittest
import numpy as np
from llm_opt.core.performance_report import PerformanceReport


class TestPerformanceReport(unittest.TestCase):
    """Test the PerformanceReport class."""

    def setUp(self):
        """Set up test fixtures."""
        self.performance_report = PerformanceReport()
        
        # Add some deterministic data
        self.c_times = [0.002, 0.001, 0.0015, 0.0025, 0.003]
        self.numpy_times = [0.006, 0.005, 0.0055, 0.0065, 0.007]
        
        for c_time in self.c_times:
            self.performance_report.add_c_runtime(c_time)
            
        for numpy_time in self.numpy_times:
            self.performance_report.add_numpy_runtime(numpy_time)

    def test_add_runtimes(self):
        """Test adding runtime measurements."""
        report = PerformanceReport()
        self.assertEqual(len(report.c_times), 0)
        self.assertEqual(len(report.numpy_times), 0)
        
        report.add_c_runtime(0.001)
        report.add_numpy_runtime(0.005)
        
        self.assertEqual(len(report.c_times), 1)
        self.assertEqual(len(report.numpy_times), 1)
        self.assertEqual(report.c_times[0], 0.001)
        self.assertEqual(report.numpy_times[0], 0.005)

    def test_calculate_c_quantiles(self):
        """Test calculating quantiles for C runtimes."""
        quantiles = self.performance_report.calculate_c_quantiles()
        
        # Check that all expected quantiles are present
        for q in [0.025, 0.25, 0.5, 0.75, 0.975]:
            self.assertIn(q, quantiles)
            
        # Check median value specifically
        self.assertAlmostEqual(quantiles[0.5], np.median(self.c_times))

    def test_calculate_numpy_quantiles(self):
        """Test calculating quantiles for NumPy runtimes."""
        quantiles = self.performance_report.calculate_numpy_quantiles()
        
        # Check that all expected quantiles are present
        for q in [0.025, 0.25, 0.5, 0.75, 0.975]:
            self.assertIn(q, quantiles)
            
        # Check median value specifically
        self.assertAlmostEqual(quantiles[0.5], np.median(self.numpy_times))

    def test_speedup_medians(self):
        """Test calculating speedup using medians."""
        expected_speedup = np.median(self.numpy_times) / np.median(self.c_times)
        actual_speedup = self.performance_report.speedup_medians()
        
        self.assertAlmostEqual(actual_speedup, expected_speedup)
        # In our test data, numpy should be slower than C
        self.assertGreater(actual_speedup, 1.0)

    def test_median_c_runtime(self):
        """Test getting median C runtime."""
        expected_median = np.median(self.c_times)
        actual_median = self.performance_report.median_c_runtime()
        
        self.assertAlmostEqual(actual_median, expected_median)

    def test_median_numpy_runtime(self):
        """Test getting median NumPy runtime."""
        expected_median = np.median(self.numpy_times)
        actual_median = self.performance_report.median_numpy_runtime()
        
        self.assertAlmostEqual(actual_median, expected_median)

    def test_desc(self):
        """Test generating performance description."""
        desc = self.performance_report.desc()
        
        # Verify basic content is present
        self.assertIn("Performance Summary", desc)
        self.assertIn("Speedup:", desc)
        self.assertIn("C Implementation", desc)
        self.assertIn("NumPy Implementation", desc)
        
        # Verify speedup value is present
        expected_speedup = np.median(self.numpy_times) / np.median(self.c_times)
        speedup_str = f"Speedup: {expected_speedup:.2f}x"
        self.assertIn(speedup_str, desc)
        
        # Verify millisecond conversion
        c_ms = np.median(self.c_times) * 1000
        numpy_ms = np.median(self.numpy_times) * 1000
        self.assertIn(f"{c_ms:.4f} ms", desc)
        self.assertIn(f"{numpy_ms:.4f} ms", desc)
        
    def test_custom_quantiles(self):
        """Test using custom quantiles."""
        custom_quantiles = [0.1, 0.9]
        report = PerformanceReport(quantiles=custom_quantiles)
        
        for c_time in self.c_times:
            report.add_c_runtime(c_time)
            
        for numpy_time in self.numpy_times:
            report.add_numpy_runtime(numpy_time)
            
        c_quantiles = report.calculate_c_quantiles()
        numpy_quantiles = report.calculate_numpy_quantiles()
        
        # Check that custom quantiles are present
        for q in custom_quantiles:
            self.assertIn(q, c_quantiles)
            self.assertIn(q, numpy_quantiles)
            
        # And default quantiles are not present
        self.assertNotIn(0.25, c_quantiles)
        self.assertNotIn(0.75, c_quantiles)


if __name__ == "__main__":
    unittest.main()