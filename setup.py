
from setuptools import setup, find_packages

setup(
    name='myocr',
    version='0.1.1',
    packages=find_packages(),
    description='A sample myocr package',
    author='Your Name',
    author_email='you@example.com',
    url='https://github.com/yourusername/myocr',
    install_requires=[
        'jupyterlab==4.2.5',
        'opencv-python==4.10.0.84',
        'matplotlib==3.9.2',
        'pandas==2.2.3',
        'tqdm==4.66.5',
        'torch==2.4.1+cpu',
        'torchvision==0.19.1+cpu',
        'pandas==2.2.3',
        
    ],
)
