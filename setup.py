from setuptools import setup


setup(
    name='nonogram_qr',
    version='0.1',
    description=(
        'Create Nonograms encoding QR codes encoding secrets'
    ),
    author='Matthew Parker',
    packages=[
        'nonogram_qr',
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'qrcode'
    ],
)