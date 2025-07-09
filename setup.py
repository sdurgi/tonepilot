from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tonepilot",
    version="0.1.0",
    author="Srivani Durgi",
    author_email="",  # Add your email if you want
    description="An emotional intelligence system for text generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tonepilot",  # Update with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "google-generativeai>=0.3.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0.0",
    ],
    include_package_data=True,
    package_data={
        'tonepilot': ['config/*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'tonepilot=tonepilot.cli.cli:main',
        ],
    },
) 