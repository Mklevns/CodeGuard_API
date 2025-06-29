"""Setup configuration for CodeGuard API package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="codeguard-api",
    version="1.0.0",
    author="CodeGuard Team",
    author_email="team@codeguard.dev",
    description="Static code analysis platform for ML/RL projects with AI-powered improvements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codeguard/codeguard-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "pydantic>=2.11.0",
        "flake8>=7.3.0",
        "pylint>=3.3.0",
        "mypy>=1.13.0",
        "black>=24.8.0",
        "isort>=5.13.0",
        "libcst>=1.5.0",
        "openai>=1.58.0",
        "anthropic>=0.42.0",
        "google-generativeai>=0.8.0",
        "psycopg2-binary>=2.9.0",
        "pyyaml>=6.0.0",
        "requests>=2.32.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.25.0",
            "pytest-cov>=6.0.0",
            "httpx>=0.28.0",
        ],
        "security": [
            "cryptography>=44.0.0",
            "python-jose[cryptography]>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "codeguard=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["rules/*.json", "static/*", "docs/*"],
    },
)