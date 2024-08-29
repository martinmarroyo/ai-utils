from setuptools import setup, find_packages

setup(
    name="aiutils",
    version="0.0.11",
    description="A set of utility functions for AI projects",
    author="Martin Arroyo",
    author_email="martinm.arroyo7@gmail.com",
    url="https://github.com/martinmarroyo/ai-utils",
    packages=find_packages(),
    install_requires=[
        "rank_bm25",
        "pydantic>=2.8.2",
        "aiohttp>=3.10.5",
        "requests>=2.32.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
