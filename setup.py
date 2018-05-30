from setuptools import setup, find_packages


setup(
    name='brokkr',
    version="0.0.1",
    packages=find_packages(),
    install_requires=['numpy'],

    # metadata for upload to PyPI
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Case study for Pytorch and Nvidia with V100 GPUs",
    license="MIT",
    keywords="profiling, pytorch, v100",
    url="https://github.com/yngtodd/nvidia",
)
