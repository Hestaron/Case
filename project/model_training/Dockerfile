# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim-buster

EXPOSE 8080

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
WORKDIR /app/
COPY ./project/requirements.txt /app/project/requirements.txt
RUN python -m pip install -r /app/project/requirements.txt
# && pip uninstall matplotlib graphviz plotly -y

# Install the functions and SQL queries in the right folder
WORKDIR /app/
COPY ./project/model_training/train_model.py ./project/model_training/train_model.py


# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
WORKDIR /app/
CMD ["python", "./project/model_training/train_model.py"]
