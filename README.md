# ML Model Deployment Assignment

This project implements a simple machine learning model deployment using FastAPI for the AI 350 Data Science course.

## Project Structure
- `data/`: Contains the synthetic dataset (`data.csv`).
- `models/`: Contains the trained model file (`model.pkl`).
- `scripts/`: Contains Python scripts for data generation and model training.
- `main.py`: The FastAPI application script.
- `requirements.txt`: List of Python dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate data and train the model:
   ```bash
   python scripts/generate_data.py
   python scripts/train_model.py
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn main.py --reload
   ```
4. Access the API documentation at `http://127.0.0.1:8000/docs`.

## Deployment (Railway)
1. Push this project to a GitHub repository.
2. Link your GitHub account to [Railway.app](https://railway.app/).
3. Create a new project and select your repository.
4. Railway will automatically detect the `Procfile` and `requirements.txt` and start the deployment.
5. Once deployed, Railway will provide a public URL for your API.
