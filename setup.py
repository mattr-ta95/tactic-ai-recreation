from setuptools import setup, find_packages

setup(
    name="tacticai",
    version="0.1.0",
    description="Recreation of DeepMind's TacticAI for soccer tactical analysis",
    author="TacticAI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "statsbombpy>=1.11.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "mplsoccer>=1.1.0",
        "streamlit>=1.25.0",
        "requests>=2.31.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "httpx>=0.24.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
    },
)
