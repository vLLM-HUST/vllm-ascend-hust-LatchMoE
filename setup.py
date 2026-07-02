from setuptools import setup, find_packages

setup(
    name="vllm-moe-offload-ascend",
    version="0.1.0",
    description="MoE Expert Offloading plugin for vllm-ascend (Ascend NPU)",
    packages=find_packages(),
    python_requires=">=3.10",
    # vLLM and the hook-enabled vllm-ascend-hust fork are external
    # prerequisites. Do not depend on the PyPI vllm-ascend package here,
    # because it does not contain the MoE offload hook seam required by this
    # plugin.
    install_requires=[],
    entry_points={
        "vllm.platform_plugins": [
            "moe_offload_ascend = vllm_moe_offload_ascend:register",
        ],
    },
)
