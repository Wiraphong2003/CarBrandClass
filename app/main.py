from fastapi import FastAPI,Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import pickle
from code import predictcar
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)
class Item(BaseModel):
    img: str 

PATH_HOG = 'http://172.17.0.2:80/api/gethog'

HOG_API_URL_DEFAULT = 'http://localhost:8080/api/gethog'
HOG_API_URL_ALTERNATE = 'http://172.17.0.2:80/api/gethog'
HEADERS = {"Content-Type": "application/json"}

m = pickle.load(open(os.getcwd()+r'/model/image_modelv3.pk', 'rb'))

# pwd == work

@app.get("/")
def root():
    return {"message": "This is my api imageCAR"}    

@app.post("/api/carbrand")
async def genhog(request: Request):
    try:
        data = await request.json()
        jsons = {"img": data['img']}

        async with aiohttp.ClientSession() as session:
            async with session.get(PATH_HOG, json=jsons, headers=HEADERS) as response:
                hog_result = await response.json()

        res = predictcar(m,[hog_result['Hog']])
        return {"predict":res}
    except:
        raise HTTPException(status_code=500, detail="invalid value")