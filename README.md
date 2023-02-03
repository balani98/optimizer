WINDOWS :

1. Steps to install Black in VS code : https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0#:~:text=Black%20is%20%22the%20uncompromising%20Python,save%20a%20file%20in%20VSCode.&text=Open%20your%20VSCode%20settings%2C%20by,%3E%20Preferences%20%2D%3E%20Settings'.&text=Black%20will%20now%20format%20your%20code%20whenever%20you%20save%20a%20*.

2. For Flake8 Linting explanation :
   the 4th extension (Linter ) he installs is Flake8 : https://www.youtube.com/watch?v=Z3i04RoI9Fk

pip install pipenv
pipenv install reqests
python --version
exit (not exit()) to come out of pipenv
pipenv shell
pipenv check
pipenv graph

Updating python version

1. Change the version on python in pipfile ex :
   Please install that version from python webpage: https://www.python.org/downloads/ (The requiured version)
   [requires]
   python_version = "3.10"
   in the shell run : pipenv --python 3.10

2. pipenv install django

3. Create a .env file to store all the secrect keys and add all of them in .gitignore files.

To check the version in your virtual environment :
pipenv run python
import sys
sys.executable
exit()

out side the environment

pipenv run python
import sys
sys.executable
exit()

other imp commands :
pipenv --venv

Only while pushing to production or when all the testing is done with version present in pipenv:
pipenv lock
pipenv install --ignore-pipfile

Push an existing folder
cd existing_folder
git init --initial-branch=main
git remote add origin https://gitlab.com/yogesh-nabler/cross_media_optimizer.git
git add .
git commit -m "Initial commit"
git push -u origin main

Push an existing Git repository
cd existing_repo
git remote rename origin old-origin
git remote add origin https://gitlab.com/yogesh-nabler/cross_media_optimizer.git
git push -u origin --all
git push -u origin --tags
