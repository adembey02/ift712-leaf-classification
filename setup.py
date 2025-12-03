from setuptools import setup, find_packages

setup(
    name='ift712-classification-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A classification project for IFT712 course using scikit-learn',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)