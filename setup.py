from setuptools import setup, find_packages

setup(
    name="penalty_shootout",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26.0",
        "gymnasium>=0.29.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.2",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "streamlit>=1.24.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning project for football penalty shootout simulation",
    url="https://github.com/yourusername/penalty-shootout-rl",
    entry_points={
        "console_scripts": [
            "train-penalty=src.cli:train_cli",
            "eval-penalty=src.cli:eval_cli",
        ],
    },
)
