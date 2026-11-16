from setuptools import setup, find_packages
setup(
    name='colabdesign',
    # A modified fork, not upstream 1.1.2: an installed copy should not claim
    # to be the PyPI package it shadows.
    version='1.1.2+germinal',
    description='Making Protein Design accessible to all via Google Colab! '
                '(Germinal fork)',
    long_description="Making Protein Design accessible to all via Google Colab!",
    long_description_content_type='text/markdown',
    packages=find_packages(include=['colabdesign*']),
    install_requires=['py3Dmol','absl-py','biopython',
                      'chex','dm-haiku','dm-tree<=0.1.9',
                      'immutabledict','jax','ml-collections',
                      'numpy','pandas','scipy','optax','joblib',
                      'matplotlib', 'iglm'],
    include_package_data=True
)
