"""Setup configuration for installing the package."""
from setuptools import find_packages
from distutils.core import setup

setup(
    name="engineai_rl_workspace",
    version="1.0.0",
    author="Darrell",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="info@engineai.com.cn",
    description="EngineAI RL workspace integrating environments and RL algorithms",
    install_requires=[
        "engineai_rl",
        "engineai_gym",
        "engineai_rl_lib",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        "numpy",
        "astor",
        "matplotlib",
        "pygame",
        "MNN",
        "onnx",
        "redis",
        "pre-commit",
    ],
)
