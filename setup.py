from setuptools import find_packages, setup

setup(
    name='bert-text-classification',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'pandas',
        'torch',
        'scikit-learn',
        'transformers'
    ],
    entry_points={
        'console_scripts': [
            'train_bert=bert_text_classification.main:main',
        ],
    },
)
