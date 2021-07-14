from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from transformers import pipeline
from prometheus_client import Histogram, Counter
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = "mlops"
SUBSYSTEM = "model"

instrumentator = Instrumentator()
instrumentator.add(metrics.request_size(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(metrics.response_size(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(metrics.latency(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(metrics.requests(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))

def model_output(metric_namespace: str = "", metric_subsystem: str = ""):
    SCORE = Histogram(
        "model_score",
        "Predicted score of model",
        buckets=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9),
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )
    SENTIMENT = Counter(
        "sentiment",
        "Predicted sentiment",
        namespace=metric_namespace,
        subsystem=metric_subsystem,
        labelnames=("sentiment",)        
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            model_score = info.response.headers.get("X-model-score")
            model_sentiment = info.response.headers.get("X-model-sentiment")
            if model_score:
                SCORE.observe(float(model_score))
                SENTIMENT.labels(model_sentiment).inc()

    return instrumentation

instrumentator.add(model_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))

app = FastAPI(
    title="Sentiment Model API",
    description="Sentiment Model API",
    version="0.1",
)

# Prometheus Instrumentator verknüpfen
instrumentator.instrument(app).expose(app)

model_path = "models/model"
sentiment_classifier = pipeline("sentiment-analysis", model_path)

class Input(BaseModel):
    sentence: str = Field(example="This is a great reddit post!")

# Datenmodell für Ausgabe
class Sentiment(BaseModel):
    label: str = Field(description="Sentiment", example="NEGATIVE")
    score: float = Field(description="Score", example=0.95)

# Endpunkt für Prediction
@app.post('/predict', response_model=Sentiment, operation_id="predict_post")
async def predict(response: Response, input: Input):
    pred = sentiment_classifier(input.sentence)[0]
    sentiment = Sentiment(**pred)
    response.headers["X-model-score"] = str(sentiment.score)
    response.headers["X-model-sentiment"] = str(sentiment.label)
    return sentiment


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)