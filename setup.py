from setuptools import setup, find_packages


setup(
    name='photoprop',
    author='T. Hoinka',
    author_email='tobias.hoinka@tu-dortmund.de',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'matplotlib>=2'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
