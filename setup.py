"""Setup script for Real-Time ML Inference Platform"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="real-time-ml-inference-platform",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade real-time ML inference platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/real-time-ml-inference-platform",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.0",
        "redis[hiredis]>=5.0.1",
        "kafka-python>=2.0.2",
        "mlflow>=2.10.0",
        "boto3>=1.34.0",
        "psycopg2-binary>=2.9.9",
        "sqlalchemy>=2.0.25",
        "prometheus-client>=0.19.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.1",
        "httpx>=0.26.0",
        "tenacity>=8.2.3",
        "pyyaml>=6.0.1",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "onnxruntime>=1.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "ruff>=0.1.14",
            "mypy>=1.8.0",
            "black>=24.1.0",
            "pre-commit>=3.6.0",
            "hypothesis>=6.98.0",
            "locust>=2.20.0",
        ],
    },
)
