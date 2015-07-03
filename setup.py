import setuptools

setuptools.setup(
    name="vtln",
    version="0.1.0",
    url="https://github.com/mwv/vtln",

    author="Maarten Versteegh",
    author_email="maartenversteegh@gmail.com",

    description="simple vtln",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
)
