from setuptools import find_packages, setup

setup(
    name="engineai_rl_lib",
    version="1.0.0",
    packages=find_packages(),
    author="Shiqin (Darrell) Dai",
    maintainer="Shiqin (Darrell) Dai",
    maintainer_email="daisq@engineai.com.cn",
    license="BSD-3",
    description="Lib for engineai_gym, engineai_rl and engineai_rl_workspace",
    python_requires=">=3.6",
    install_requires=["GitPython", "pybullet", "pandas"],
)
