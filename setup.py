import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="heimr", # Replace with your own username

    version="0.0.1",
    author="Raphael Memmesheimer",
    author_email="raphael@uni-koblenz.de",
    description="Simitate: A Hybrid Imitation Learning Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/airglow/simitate",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
