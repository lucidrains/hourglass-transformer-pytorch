from setuptools import setup, find_packages

setup(
  name = 'hourglass-transformer-pytorch',
  packages = find_packages(),
  version = '0.0.7',
  license='MIT',
  description = 'Hourglass Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/hourglass-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'einops',
    'torch>=1.6',
    'rotary-embedding-torch',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
