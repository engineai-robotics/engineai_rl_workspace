from setuptools import find_packages, setup

setup(
    name="engineai_rl",
    version="1.0.0",
    packages=find_packages(),
    author="Shiqin (Darrell) Dai",
    maintainer="Shiqin (Darrell) Dai",
    maintainer_email="daisq@engineai.com.cn",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=["tensorboard", "wandb"],
)
