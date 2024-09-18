from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ProductObject(BaseModel):
    product1: str
    product2:str

class Similarity_Search:
    
    def check_similarity(self,product1,product2):
        import pandas as pd
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

        product_embedding = model.encode(product1, convert_to_tensor=True)
        claim_embedding = model.encode(product2, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(product_embedding, claim_embedding).item()
        infringement = "Yes" if similarity_score > 0.7 else "No"  # Adjust threshold as needed
        return infringement
    
@app.post("/check_simi/")
async def generate_claim_chart_endpoint(input: ProductObject):
    try:
        checking_result=Similarity_Search().check_similarity(input.product1,input.product2)
        return {"claim_chart": checking_result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
