# PhishShift — Liquid Glass Active Dashboard

A beginner-friendly Streamlit project for educational message-risk triage. Includes a live session scan table, risk score percentages, model comparison, dataset visualizations, and CSV export.

## Run it (Windows)
1. Extract the ZIP.
2. Open the extracted `PhishShift_Liquid_Glass` folder.
3. Double-click `setup.bat` on the first run. It creates `.venv`, installs dependencies, trains models, and launches the dashboard.
4. Later, double-click `run.bat`.
5. Browser address is normally `http://localhost:8501`.

You can also open this folder in VS Code and use Terminal:
```powershell
.\setup.bat
```

## Project structure
- `app.py` — Streamlit dashboard and scanner
- `src/pipeline.py` — cleaning, models, evaluation, inference
- `data/raw/messages.csv` — generated synthetic examples
- `reports/` — model evaluation outputs after setup
- `models/` — trained model file after setup
- `tests/` — manual test plan

## Important limitations
The included dataset is synthetic and generated for demonstration. Scores are not calibrated probabilities and are not real-world guarantees. For an academic submission, document permitted external datasets and evaluate on unseen campaigns and legitimate urgent messages. Do not enter real credentials or private messages. The app never opens or fetches links.
