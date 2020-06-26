import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ML_from_scratch",
    version="1.0.0",
    author="Jørgen Navjord",
    author_email="navjordj@gmail.com",
    description="Machine learning algorithms implemented using sklearn API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/navjordj/ML_from_scratch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
