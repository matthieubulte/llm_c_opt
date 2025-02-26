import platform
import subprocess
import os
import json
import time
import numpy as np
from datetime import datetime


def get_sysctl_info():
    """Get system information using sysctl on macOS"""
    sysctl_mappings = {
        "hw.ncpu": "cpu_count",
        "hw.physicalcpu": "physical_cpu_count",
        "hw.memsize": "memory_total",
        "hw.l1icachesize": "l1i_cache",
        "hw.l1dcachesize": "l1d_cache",
        "hw.l2cachesize": "l2_cache",
        "hw.l3cachesize": "l3_cache",
        "hw.cpufamily": "cpu_family",
        "hw.cputype": "cpu_type",
        "hw.cpusubtype": "cpu_subtype",
        "hw.cpu64bitcapable": "cpu_64bit_capable",
        "hw.cpufrequency": "cpu_frequency",
        "hw.busfrequency": "bus_frequency",
        "hw.tbfrequency": "tb_frequency",
        "hw.packages": "packages",
        "hw.logicalcpu_max": "logical_cpu_max",
        "hw.physicalcpu_max": "physical_cpu_max",
    }

    result = {}
    for sysctl_key, result_key in sysctl_mappings.items():
        try:
            value = (
                subprocess.check_output(["sysctl", sysctl_key])
                .decode()
                .strip()
                .split(": ")[1]
            )
            # Try to convert to int if possible
            try:
                value = int(value)
            except ValueError:
                pass
            result[result_key] = value
        except:
            pass

    return result


def get_mac_cpu_info():
    """Get detailed CPU info for Mac M1"""
    try:
        # Using system_profiler to get CPU information
        cpu_info = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"]
        ).decode()

        # Parse the output
        info = {}
        for line in cpu_info.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = [x.strip() for x in line.split(":", 1)]
                info[key] = value

        return info
    except Exception as e:
        return {"error": str(e)}


def get_compiler_info():
    """Get compiler information"""
    compiler_info = {}

    try:
        # Get clang version (default on macOS)
        clang_version = (
            subprocess.check_output("clang --version", shell=True)
            .decode()
            .split("\n")[0]
        )
        compiler_info["clang"] = clang_version
    except:
        compiler_info["clang"] = "Unknown"

    try:
        # Get GCC version if installed
        gcc_version = (
            subprocess.check_output("gcc --version", shell=True).decode().split("\n")[0]
        )
        compiler_info["gcc"] = gcc_version
    except:
        compiler_info["gcc"] = "Not installed or not in PATH"

    return compiler_info


def get_memory_latency_estimate():
    """Simple memory latency benchmark"""
    try:
        array_size = 10000000
        arr = np.random.rand(array_size)

        iterations = 10000
        start = time.time()
        for _ in range(iterations):
            idx = np.random.randint(0, array_size)
            val = arr[idx]  # Force memory access
        end = time.time()

        latency_ns = ((end - start) * 1e9) / iterations
        return {"random_access_latency_ns": latency_ns}
    except Exception as e:
        return {"error": str(e)}


def run_simd_benchmark():
    """Measure SIMD/vector performance with NumPy"""
    try:
        # Vector addition benchmark
        size = 20000000  # Large enough to be meaningful

        # Create large arrays
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)

        # Measure vector addition time
        start = time.time()
        c = a + b
        vector_time = time.time() - start

        # Scalar benchmark for comparison
        start = time.time()
        c_scalar = np.zeros(size, dtype=np.float32)
        for i in range(min(100000, size)):  # Just do a subset for scalar
            c_scalar[i] = a[i] + b[i]
        scalar_time_sample = time.time() - start

        # Extrapolate to full size
        scalar_time_extrapolated = scalar_time_sample * (size / min(100000, size))

        # Calculate speedup
        estimated_speedup = (
            scalar_time_extrapolated / vector_time if vector_time > 0 else 0
        )

        return {
            "vector_time_seconds": vector_time,
            "scalar_time_extrapolated_seconds": scalar_time_extrapolated,
            "estimated_vector_speedup": estimated_speedup,
        }
    except Exception as e:
        return {"error": str(e)}


def run_memory_bandwidth_test():
    """Measure memory bandwidth"""
    try:
        # Create large array to ensure we're testing memory not cache
        size = 50000000  # ~200MB for float32
        iterations = 5

        # Memory read benchmark
        a = np.random.rand(size).astype(np.float32)
        start = time.time()
        for _ in range(iterations):
            b = np.sum(a)  # Force memory read of entire array
        end = time.time()

        # Calculate bandwidth
        bytes_read = size * 4 * iterations  # float32 = 4 bytes
        seconds = end - start
        bandwidth_gbps = (bytes_read / 1e9) / seconds

        return {"memory_bandwidth_gbps": bandwidth_gbps}
    except Exception as e:
        return {"error": str(e)}


def get_apple_silicon_features():
    """Get Apple Silicon specific features if possible"""
    try:
        # Check if it's Apple Silicon
        machine = platform.machine()
        if machine not in ["arm64", "aarch64"]:
            return {"is_apple_silicon": False}

        # Detect specific chip by parsing model identifier
        model_info = subprocess.check_output(["sysctl", "hw.model"]).decode().strip()

        # Get more detailed information using ioreg
        ioreg_output = subprocess.check_output(
            ["ioreg", "-c", "IOPlatformDevice", "-d", "2"]
        ).decode()

        # Try to extract performance core and efficiency core counts
        result = {
            "is_apple_silicon": True,
            "model_info": model_info,
        }

        # Try to identify specific chip (M1, M1 Pro, M1 Max, M2, etc.)
        if "Macmini" in model_info and "2020" in ioreg_output:
            result["chip_type"] = "M1"
        elif "MacBookPro" in model_info and "2021" in ioreg_output:
            if "M1 Max" in ioreg_output:
                result["chip_type"] = "M1 Max"
            elif "M1 Pro" in ioreg_output:
                result["chip_type"] = "M1 Pro"
            else:
                result["chip_type"] = "M1"
        elif "MacBookAir" in model_info and "2020" in ioreg_output:
            result["chip_type"] = "M1"
        elif "Mac" in model_info and "2022" in ioreg_output:
            result["chip_type"] = "M2"
        else:
            # Default to detecting M1 if we can't be more specific
            result["chip_type"] = "M1 family"

        return result
    except Exception as e:
        return {"error": str(e), "is_apple_silicon": True}


def get_full_system_info():
    """Gather comprehensive system information"""
    system_info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "mac_cpu_info": get_mac_cpu_info(),
        "sysctl_info": get_sysctl_info(),
        "compilers": get_compiler_info(),
        "memory_latency": get_memory_latency_estimate(),
        "simd_performance": run_simd_benchmark(),
        "memory_bandwidth": run_memory_bandwidth_test(),
        "apple_silicon": get_apple_silicon_features(),
    }

    return system_info


if __name__ == "__main__":
    # Check if we're on macOS
    if platform.system() != "Darwin":
        print(
            "This script is optimized for macOS. Some information may not be available."
        )

    # Get system info
    system_info = get_full_system_info()

    # Print as formatted JSON
    print(json.dumps(system_info, indent=2))

    # Save to file
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)

    print(f"System information saved to 'system_info.json'")
    print(
        "You can provide this file to an LLM to help generate optimized C code for your M1 Mac."
    )
