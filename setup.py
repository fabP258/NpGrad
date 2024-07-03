from setuptools import setup, find_packages

setup(
    name="npgrad",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "pytest"],
    author="Your Name",
    description="A neural network library with automatic differentiation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fabP258/NpGrad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
