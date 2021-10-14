from setuptools import setup

setup(
    name='pydale',
    version='0.1.0a1',
    description='A Domain Adaptation Python package',
    url='https://github.com/sz144/pydale',
    author='Shuo Zhou',
    author_email='shuo.zhou@sheffield.ac.uk',
    license='MIT License',
    packages=['pydale'],
    install_requires=['numpy', 'scipy', 'pandas',
                      'scikit-learn', 'cvxopt', 'osqp',
                      'pytest', 'pykale'
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
