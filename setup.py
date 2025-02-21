from setuptools import setup, find_packages

setup(
    name="f1-llm",
    version="0.1",
    packages=find_packages(),
    # install_requires=[
    #     "tkinter",
    # ],
    entry_points={
        "console_scripts": [
            "f1-llm = f1_llm.main:main",
        ],
    },
)