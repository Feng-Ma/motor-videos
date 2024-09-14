from setuptools import setup, find_packages

setup(
    name="motor-videos",
    version="1.0.0",
    author="Feng Ma",
    author_email="fengma@ucm.es",
    description="Motor de videos para el pipeline del TFM de UCM",
    long_description="Motor de videos para el pipeline del TFM de UCM",
    long_description_content_type="text/markdown",
    url="https://github.com/Feng-Ma/motor_videos",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=['loguru']
)
