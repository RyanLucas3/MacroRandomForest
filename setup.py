from distutils.core import setup

setup(
    # How you named your package folder (MyLib)
    name='MacroRandomForest',
    packages=['MacroRandomForest'],   # Chose the same as "name"
    # Start with a small number and increase it with every change you make
    version='1.0.3',
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='MIT',
    # Give a short description about your library
    description='Macroeconomic Random Forest by Ryan Lucas (code) and Philippe Goulet Coulombe (method)',
    author='',                   # Type in your name
    author_email='ryanlu@mit.edu',      # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://github.com/RyanLucas3/MacroRandomForest',
    # I explain this later on
    download_url='https://github.com/RyanLucas3/MacroRandomForest/archive/refs/tags/1.0.3.tar.gz',
    # Keywords that define your package best
    keywords=['Time Series', 'Forecasting', 'Economics',
              "Macro", "Machine Learning", "RandomForest"],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'joblib',
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
