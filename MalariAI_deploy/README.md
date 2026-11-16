# MalariAI prototype

A Streamlit prototype for AI-assisted thick-smear image screening, surveillance overview, and comparison with microscopist readings.

## Run locally

1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run dashboard.py`

The trained model is expected at `models/best.pt`.

## Deploy

Push this folder to a GitHub repository, then create an app at https://share.streamlit.io and select `dashboard.py` as the app entrypoint.

## Important prototype limitations

- The app contains synthetic demo records and illustrative district positivity values. Do not present those as real national surveillance data.
- The app writes its SQLite database and uploaded smear images to its local filesystem. This is not a durable production database or secure clinical image store.
- Do not upload identifiable patient data or real clinical images to a public prototype deployment.
- AI outputs are screening support only and require qualified microscopist confirmation. This prototype is not a validated diagnostic device.
