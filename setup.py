"""
$QUANTAGENT - Setup
"""
from setuptools import setup, find_packages

setup(
    name="quantagent",
    version="1.0.0",
    author="QuantAgent Team",
    description="Elite Quantitative Trading Analysis in One Command",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/quantagent/quantagent",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "yfinance>=0.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
    ],
    entry_points={
        "console_scripts": [
            "quantagent=quantagent.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
