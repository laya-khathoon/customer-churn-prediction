# Customer Churn Prediction on Vercel

The Vercel version uses a static, mobile-friendly web page and a stateless Python function. It does not try to run Streamlit inside a serverless function.

## Why no Streamlit or XGBoost install

Vercel's free serverless functions have a size limit (250 MB unzipped). The `xgboost` library alone installs at well over that once its native dependencies are included, so the usual `pip install -r requirements.txt` approach cannot fit. Instead, `scripts/export_web_model.py` converts the trained model in `models/churn_model.pkl` into `api/model_web.json` (200 decision trees as JSON, about 360 KB), and `web_model.py` evaluates it with only the Python standard library. `tests/test_web_model.py` proves the JSON predictor matches the real pipeline on all 7,043 dataset rows plus edge cases.

## Files

- `public/index.html`: customer profile, services and billing form with the prediction result.
- `api/index.py`: stateless prediction function (`POST /api`), no third-party packages.
- `api/model_web.json`: the exported model. Regenerate with `python scripts/export_web_model.py` after retraining.
- `web_model.py`: dependency-free predictor shared by the function and the tests.
- `vercel.json`: static output, function duration and security headers.
- The original `app.py`, `src/`, `models/churn_model.pkl` and tests remain in place for the existing Streamlit app, unchanged.

## Deploy again

1. Import `laya-khathoon/customer-churn-prediction` in Vercel, using the free Hobby team.
2. Use production branch `main`, root directory `./` and Application Preset `Other`.
3. The repository configuration sets the output directory to `public`. Leave the build command empty.
4. No environment variables or API keys are needed.
5. Deploy, then open the production URL, fill the form and confirm a prediction renders.

## Regenerate the web model after retraining

```bash
python src/train.py
python scripts/export_web_model.py
python -m pytest tests/
```

## What the results mean

- The probability comes from the same trained XGBoost model as the Streamlit app, evaluated from its JSON export.
- A customer at or above the decision threshold (0.40) is labelled likely to churn. The risk bands are display labels, not guarantees.
- The form values are sent only to this app's own function for scoring; nothing is saved to a database or browser storage.

## Sync the Windows git clone

```powershell
cd <your customer-churn-prediction folder>
git status
git pull --ff-only origin main
```

If `git status` shows local work, keep it safe before pulling.

The Streamlit app still runs with `python -m streamlit run app.py`. To run the Vercel version locally, use Vercel CLI `vercel dev` from the repository root. Vercel free-tier limits still apply; no paid services are required by this implementation.
