from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cnap",
    version="0.1",
    author="heyu",
    author_email="yh441@cam.ac.uk",
    description="Continuous Neural Algorithmic Planner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dransyhe/part-ii-project",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'classic = cnap.training.classic_control:main',
            'maze = cnap.training.maze:main',
            'mujoco = cnap.training.mujoco:main',
        ],
    },
    install_requires=[
        "gym<0.20,>=0.17",
        "matplotlib>=3.3.4",
        "networkx>=2.5",
        "numpy>=1.19.5",
        "opencv_python>=4.5.1.48",
        "pandas>=1.3.4",
        "scikit_learn>=1.0.2",
        "seaborn>=0.11.2",
        "stable_baselines3>=1.3.0",
        "torch>=1.9.1",
        "torch_scatter>=2.0.9",
    ],
)

