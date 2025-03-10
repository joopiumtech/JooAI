from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from auth import auth_test, get_current_user
from merchant_backend import query_db_for_merchant


# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# Initialize FastAPI app
app = FastAPI()

# ----------------------------------------------------------------
# Merchant APIs
# ----------------------------------------------------------------


# Authentication API
class MerchantAuthRequest(BaseModel):
    email: str
    password: str


class MerchantAuthResponse(BaseModel):
    is_authenticated: bool
    auth_message: str
    access_token: str = None  # Include token in the response


@app.post("/auth", response_model=MerchantAuthResponse)
async def merchant_auth(request: MerchantAuthRequest):
    """
    FastAPI endpoint for merchant authentication.
    """
    try:
        result = auth_test(email=request.email, password=request.password)
        return result
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unexpected error: {str(e)}")


# Chat API
class MerchantQueryRequest(BaseModel):
    query: str
    audio_query: bool


class MerchantQueryResponse(BaseModel):
    ai_response: str


@app.post("/chat", response_model=MerchantQueryResponse)
async def merchant_query(
    request: MerchantQueryRequest,
    current_user: str = Depends(get_current_user),  # Require authentication
):
    """
    FastAPI endpoint to handle merchant queries.
    Only authenticated users can access.
    """
    try:
        result = query_db_for_merchant(query=request.query, audio_query=request.audio_query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



