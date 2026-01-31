# Pearls AQI Predictor

An end-to-end, serverless Air Quality Index prediction system. This project forecasts AQI for the next three days by automating data collection, feature engineering, model training, and prediction serving.

## Interesting Techniques

The implementation demonstrates several practical techniques:

* **Feature Store Architecture:** Implements a centralized feature store for managing and versioning ML features, separating data engineering from model training.
* **Scheduled Pipeline Automation:** Uses workflow orchestration to run data collection hourly and model training daily without manual intervention.
* **Model Explainability Integration:** Incorporates techniques like SHAP and LIME to make black-box model predictions interpretable.
* **Serverless Deployment:** Builds the entire system using serverless components to minimize infrastructure management.

## Technologies & Libraries

The project leverages these specific tools:

* **Data Sources:** [AQICN](https://aqicn.org/api/), [OpenWeather API](https://openweathermap.org/api)
* **Feature Store:** [Hopsworks](https://www.hopsworks.ai/) or [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore)
* **ML Frameworks:** [Scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/)
* **Model Explainability:** [SHAP](https://shap.readthedocs.io/), [LIME](https://github.com/marcotcr/lime)
* **Orchestration:** [Apache Airflow](https://airflow.apache.org/), [GitHub Actions](https://docs.github.com/en/actions)
* **Dashboard & API:** [Streamlit](https://streamlit.io/), [Gradio](https://www.gradio.app/), [FastAPI](https://fastapi.tiangolo.com/)
* **Documentation:** [Document](https://docs.google.com/document/d/1fCdXBpvRllxhjHpFGUZOcUC6sfv8qZS6sKpvcpLmhMU/edit?usp=sharing)

## Project Structure

```
pearls-aqi-predictor/
├── data_pipeline/          # Feature extraction and backfilling
├── training_pipeline/      # Model training and evaluation
├── dashboard/              # Prediction dashboard and API
├── cicd/                   # Pipeline automation configurations
├── notebooks/              # Exploratory data analysis
├── model_registry/         # Model versioning and storage
├── docs/                   # Project documentation
└── requirements.txt        # Python dependencies
```

* `data_pipeline/`: Contains scripts for fetching raw API data, computing features, and populating the feature store.
* `training_pipeline/`: Handles model training, hyperparameter tuning, and performance evaluation.
* `dashboard/`: Serves predictions through a web interface and API endpoints.
* `notebooks/`: Includes initial data exploration and model prototyping work.
