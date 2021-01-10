from setuptools import setup

setup(
    name='TPy',
    version='0.0.1',
    description='A Transfer Learning Python package',
    url='https://github.com/sz144/TPy',
    author='Shuo Zhou',
    author_email='szhou20@sheffield.ac.uk',
    license='MIT License',
    packages=['TPy'],
    install_requires=['numpy', 'scipy', 'pandas',
                      'scikit-learn', 'cvxopt', 'osqp'
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
