# UK Inclusive Wealth Dashboard

Streamlit app that builds a UK inclusive-wealth view using live data from the World Bank Wealth Accounts/Indicators and the UK ONS capital stock API. Indicators are indexed for comparison and include produced, human, natural, knowledge, and social capital.

## Run locally
1. Create/activate a virtual environment (optional but recommended).
2. Install deps: `pip install -r requirements.txt`.
3. Start the app: `streamlit run dashboard.py`.

## Deploy to Streamlit Community Cloud
1. Push this project to a **public GitHub repo** (see quick git steps below).
2. Go to https://streamlit.io/cloud → “Deploy an app”.
3. Connect your GitHub account and select the repo, branch `main`, and app file `dashboard.py`.
4. Add a Python version if prompted (tested with Python 3.11–3.13); dependencies will be installed from `requirements.txt`.
5. Deploy. The app will fetch data live from World Bank and ONS APIs—no secrets required.

## Quick git steps
```bash
git init
git add .
git commit -m "Add UK Inclusive Wealth Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

After pushing, use the Streamlit Cloud steps above and share the app URL with your professor.
