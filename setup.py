from setuptools import setup, find_packages

setup(
    name="tacticai",
    version="0.1.0",
    description="Recreation of DeepMind's TacticAI for soccer tactical analysis",
    author="TacticAI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "statsbombpy>=1.11.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
    },
)
