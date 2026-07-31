import joblib
import pandas as pd
model = joblib.load('pipeline1')
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*'])

class example(BaseModel):
  crim:float
  zn:float
  indus:float
  nox:float	
  rm:float	
  age:float	
  dis:float
  rad:float
  tax:float	
  ptratio:float	
  b:float
  lstat:float

@app.post('/predict_price')
def predict(data:example):
  house = pd.DataFrame([data.model_dump()])
  prediction = model.predict(house)
  return {'estimated_price':float(prediction)}
