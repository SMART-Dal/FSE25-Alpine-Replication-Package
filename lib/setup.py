from setuptools import setup, find_packages

setup(
    name='polp', 
    version='0.1.0',  
    author='Anonymous', 
    author_email='anon@email.domain', 
    description='A Python library for Programming Language Processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='xxx',
    packages=find_packages(where='.', exclude=['tests', 'examples']),
    install_requires=[
        'torch==2.1.2',
        'numpy==1.26.2',
        'curated_tokenizers',
        'catalogue',
        'huggingface-hub',
        'safetensors',
        'tokenizers',
    ],
    entry_points={
        'polp_encoders': [    
            'roberta = polp.nn.models.roberta:RoBERTaEncoder',
            'bert = polp.nn.models.bert:BERTEncoder',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='programming language processing, code analysis, AI for code',
)