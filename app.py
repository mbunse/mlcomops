from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from enum import Enum
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
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
    LABEL = Counter(
        "label",
        "Predicted label",
        namespace=metric_namespace,
        subsystem=metric_subsystem,
        labelnames=("label",)        
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            model_score = info.response.headers.get("X-model-score")
            model_label = info.response.headers.get("X-model-label")
            if model_score:
                SCORE.observe(float(model_score))
                LABEL.labels(model_label).inc()

    return instrumentation

instrumentator.add(model_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))

app = FastAPI(
    title="Titanic Survival Model API",
    description="Titanic Survival Model API",
    version="0.1",
)

# Prometheus Instrumentator verknüpfen
instrumentator.instrument(app).expose(app)

model_path = "models/model.pkl"
classifier = joblib.load(model_path)

class EmbarkedEnum(str, Enum):
    cherbourg = 'C'
    queenstown = 'Q'
    southampton = 'S'
    not_given = ''

class Input(BaseModel):
    pclass: int = Field(example=1, description="Passenger Class")
    name: str = Field(example="Johanna, Miss. Smith", description="Passenger Name")
    sex: str = Field(example="female", description="Sex")
    age: float = Field(example=15.0, description="Age")
    sibsp: int = Field(example=3, description="Number of Siblings/Spouses Aboard")
    parch: int = Field(example=2, description="Number of Parents/Children Aboard")
    ticket: str = Field(example="24160", description="Ticket Number")
    fare: float = Field(example=211.3375, description="Passenger Fare")
    cabin: str = Field(example="B5", description="Cabin")
    embarked: EmbarkedEnum = Field(description="Port of Embarkation")
    home_dest: str = Field(alias="home.dest", example="Montreal, PQ / Chesterville, ON", description="Heimat/Ziel")

# Datenmodell für Ausgabe
class Prediction(BaseModel):
    label: int = Field(description="Survival", example=1)
    score: float = Field(description="Score", example=0.95)

# Endpunkt für Prediction
@app.post('/predict', response_model=Prediction, operation_id="predict_post")
async def predict(response: Response, input: Input):
    df = pd.DataFrame([input.dict(by_alias=True)])
    pred_probas = classifier.predict_proba(df)[0]
    survival = np.argmax(pred_probas)
    prediction = Prediction(label=survival, score=pred_probas[survival])
    response.headers["X-model-score"] = str(prediction.score)
    response.headers["X-model-label"] = str(prediction.label)
    return prediction


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)