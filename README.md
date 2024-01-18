# The case

## General information
### Task 1 
- Build a predictive model in Python to predict product02. 
- Please take into account all stages needed to build a predictive model (data exploration etc).

Nice to have: 
- Push your source code into Git hosting platform of your choice. 
- Save your model in ML Flow. If you don't have an mlflow instance, just show us the code that would do it.

### Task 2 (Bonus)
- Create a simple Dockerfile for running your scripts

### Task 3 (Bonus)
- Schematically design a CI/CD pipeline that would validate and push your created model script/dockerfile to a hypothetical cloud environment. (No need to actually implement this, only a diagram is needed)

## Preparation

### Setting up environment + retrieve files
1. Open a terminal and go to the folder you want the files in
2. Download the files to your current folder ```git clone https://github.com/Hestaron/Case.git```
3. Go to the 'Case' directory
4. Typ ```conda create -n <environment-name> --file requirements.txt``` to set up your environment

### Precommit installeren (Optioneel)
1. Activate your environment ```conda activate <environment-name>```
2. Install pre-commit ```pip install pre-commit```
3. ```pre-commit install --hook-type pre-commit```

## The data

### Exploratory