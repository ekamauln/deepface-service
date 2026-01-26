# Deepface Attendance System (Python + FastAPI + Tensorflow)

This is a microservice for face recognition-based attendance system using Deepface library, built with Python 3.12 and FastAPI.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Python - Official](https://marketplace.visualstudio.com/items?itemName=ms-python.python) + [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) + [autopep8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8)

## Quick Start

```bash


# Clone the repo
git clone https://github.com/ekamauln/deepface-service.git

# Use Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app in development mode
fastapi dev main.py --reload

# Run the app in production mode
fastapi run main:app --host 0.0.0.0 --port 8000
```
