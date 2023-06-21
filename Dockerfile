FROM python:latest
WORKDIR /code
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt


# Creating folders, and files for a project:
COPY . .



