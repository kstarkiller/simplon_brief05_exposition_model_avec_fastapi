from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import uvicorn
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

# LOADS DES MODELES 
with open(r"model/paris_model.pkl", "rb") as f:
    my_unpickler = pickle.Unpickler(f)
    paris_model = my_unpickler.load()

with open(r"model/idf_model.pkl", "rb") as g:
    my_unpickler = pickle.Unpickler(g)
    idf_model = my_unpickler.load()


# DECLARATIONS DES ROUTES POST
@app.get("/sq2_price_predictor_PARIS/", description="Retourne une prédiction de prix au m²")
async def sq2_price_predictor_PARIS(longitude: str, latitude: str):
    input_data = np.array([[longitude, latitude]])

    return paris_model.predict(input_data)[0]

@app.get("/sq2_price_predictor_IDF/", description="Retourne une prédiction de prix au m²")
async def sq2_price_predictor_IDF(longitude: str, latitude: str, taux: str):
    input_data = np.array([[longitude, latitude, taux]])
    
    return idf_model.predict(input_data)[0]


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=5000)


uvicorn.run(app)