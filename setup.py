import setuptools

setuptools.setup(
    name="martini",
    version="0.0.1",
    author="Ignasi Puch-Giner",
    author_email="ignasi.puch.giner@gmail.com",
    description="A Python package to automatize Martini simulations.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # This includes data specified in MANIFEST.in
    package_data={
        'martini': ['ff/*', 'scripts/*'],
      },
    python_requires='>=3.6',
)
