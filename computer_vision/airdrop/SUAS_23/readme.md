# Python Computer Vision Folder for Image Detection

## Prerequisites

- Python 3.11+: Just install using the official way
- pipenv:
  - [Official Instructions](https://pipenv.pypa.io/en/latest/installation/)
  - Quick Install:
    - Universal: `pip install --user pipenv`
    - Mac (using brew if you want): `brew install pipenv`

## Usage

1. **Important**: Go to the correct directory: `cd computer_vision/airdrop/SUAS_23`
2. Install the environment(only the first time): `pipenv install && pipenv install --dev`
3. Launch the environment: `pipenv shell`
4. Use Python normally: `python <name of your script>`

After the first install, you just need to run steps 3 and 4 for running anything in the future

If you want to install any additional packages, run: `pipenv install <package name>` just like you would for pip. For example, `pip install opencv-python` is now `pipenv install opencv-python`

## Running tests

Pytest usage is documente [here](https://docs.pytest.org/en/6.2.x/usage.html)

- Run all tests: `pytest`
- With verbose output: `pytest -v`
- Run tests from a specifc file: `pytest -v test_recognition.py`
- **Note:** Parametrized test labels are generated as "[Shape-Symbol-Color]", so you can use the -k flag to select the tests you want
  - For ambiguous labels, eg "Circle" might select "Circle" and "Semi_Circle", you can use the "[" to differentiate between them
  - Circle would be labelled as: "[Circle-Symbol-Color]"
  - Semi_Circle would be labelled as: "[Semi_Circle-Symbol-Color]"
  - so "[Circle" would only pick up "[Circle-Symbol-Color]"
- Run only specific combinations of tests, for example for detection:
  - Only for the color Blue: `pytest -v test_recognition.py -k "Blue"`
  - Only for the symbol A: `pytest -v test_recognition.py -k "sym_A"`
  - Only for the shape Triangle: `pytest -v test_recognition.py -k "Triangle"`
  - Only for the shape Circle: `pytest -v test_recognition.py -k "[Circle"`
    - Note the "[" before Circle so it isnt confused with Semi_Circle
  - Blue and Semicircle : `pytest -v test_recognition.py -k "Semi_Circle and Blue"`
