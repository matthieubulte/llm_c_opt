import os
import numpy as np
import ctypes
from typing import Dict


# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL")

SYSTEM_INFO = """
{"platform":"macOS-15.2-arm64-arm-64bit","system":"Darwin","release":"24.2.0","version":"Darwin Kernel Version 24.2.0: Fri Dec  6 18:40:14 PST 2024; root:xnu-11215.61.5~2/RELEASE_ARM64_T8103","architecture":"arm64","mac_cpu_info":{"Hardware":"","Hardware Overview":"","Model Name":"MacBook Air","Model Identifier":"MacBookAir10,1","Model Number":"Z124000KMFN/A","Chip":"Apple M1","Total Number of Cores":"8 (4 performance and 4 efficiency)","Memory":"16 GB","System Firmware Version":"11881.61.3","OS Loader Version":"11881.61.3","Serial Number (system)":"C02G615LQ6LR","Hardware UUID":"84DE28CF-40C4-5ED5-A8D8-AFE5AA1D9665","Provisioning UDID":"00008103-001939162679001E","Activation Lock Status":"Enabled"},"sysctl_info":{"cpu_count":8,"physical_cpu_count":8,"memory_total":17179869184,"l1i_cache":131072,"l1d_cache":65536,"l2_cache":4194304,"cpu_family":458787763,"cpu_type":16777228,"cpu_subtype":2,"tb_frequency":24000000,"packages":1,"logical_cpu_max":8,"physical_cpu_max":8},"compilers":{"gcc":"Apple clang version 16.0.0 (clang-1600.0.26.6)"},"memory_latency":{"random_access_latency_ns":1681.5185546875},"simd_performance":{"vector_time_seconds":0.011857271194458008,"scalar_time_extrapolated_seconds":2.8914451599121094,"estimated_vector_speedup":243.8541813282931},"memory_bandwidth":{"memory_bandwidth_gbps":16.556811722352048},"apple_silicon":{"is_apple_silicon":true,"model_info":"hw.model: MacBookAir10,1","chip_type":"M1 family"}}
"""


# Mapping of NumPy dtypes to C types
DTYPE_TO_CTYPE_STR: Dict[type, str] = {
    np.float32: "float",
    np.float64: "double",
    np.int32: "int",
    np.int64: "long long",
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long",
    np.bool_: "bool",
}

# Mapping of NumPy dtypes to ctypes
DTYPE_TO_CTYPES: Dict[type, type] = {
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.int32: ctypes.c_int,
    np.int64: ctypes.c_longlong,
    np.uint32: ctypes.c_uint,
    np.uint64: ctypes.c_ulonglong,
    np.bool_: ctypes.c_bool,
}
