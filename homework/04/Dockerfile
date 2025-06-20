FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

# Copy the  script and Pipfiles
COPY [ "starter.py", "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# Set entrypoint to run the script
ENTRYPOINT ["python", "starter.py"]