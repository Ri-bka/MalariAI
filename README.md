# MalariAI

Malaria surveillance and AI-assisted microscopy decision-support prototype.

## Features
- Surveillance overview with illustrative synthetic demonstration data
- Geographic and temporal summaries
- AI-assisted blood-smear image screening using the trained YOLO model
- Comparison of AI screening results with microscopist readings

## Run locally

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run dashboard.py
```

## Model file

Place the trained model at `models/best.pt`.

## Important notes

- Demonstration surveillance records are synthetic and are not official Ethiopian malaria statistics.
- AI screening outputs require review by a trained microscopist and must not be used as standalone clinical diagnoses.
- The local SQLite database is suitable for prototyping, not production-grade persistent storage.
