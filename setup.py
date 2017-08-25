try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'name': 'Genetic Algorithm Learning Tool',
	'version': '0.1',
	'url': 'https://github.com/parnellj/genetic_algorithms',
	'download_url': 'https://github.com/parnellj/genetic_algorithms',
	'author': 'Justin Parnell',
	'author_email': 'parnell.justin@gmail.com',
	'maintainer': 'Justin Parnell',
	'maintainer_email': 'parnell.justin@gmail.com',
	'classifiers': [],
	'license': 'GNU GPL v3.0',
	'description': 'Experiments with genetic algorithms, following a guide from ai-junkie.com.',
	'long_description': 'Experiments with genetic algorithms, following a guide from ai-junkie.com.',
	'keywords': '',
	'install_requires': ['nose'],
	'packages': ['genetic_algorithms'],
	'scripts': []
}
	
setup(**config)
