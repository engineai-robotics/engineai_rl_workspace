from setuptools import find_packages
from distutils.core import setup

setup(
    name="engineai_gym",
    version="1.0.0",
    author="Shiqin (Darrell) Dai",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="daisq@engineai.com.cn",
    description="EngineAI RL training environment for robots",
    install_requires=["isaacgym", "moviepy"],
)
