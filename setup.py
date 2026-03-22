from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="batch-finder",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Find maximum batch size, documents, and timesteps for PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/batch-finder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "transformers": ["transformers>=4.0.0"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)

