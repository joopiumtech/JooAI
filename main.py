
import threading


from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from auth import auth_test, get_current_user
from merchant_backend import query_db_for_merchant, record_audio, recording_event

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Merchant APIs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------
# Merchant Auth API
# ----------------------------------------------------------------
class MerchantAuthRequest(BaseModel):
    email: str
    password: str


class MerchantAuthResponse(BaseModel):
    is_authenticated: bool
    auth_message: str
    access_token: str = None  # Include token in the response


@app.post("/auth", response_model=MerchantAuthResponse)
async def merchant_auth(request: MerchantAuthRequest):
    """ FastAPI endpoint for merchant authentication. """
    try:
        result = auth_test(email=request.email, password=request.password)
        return result
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unexpected error: {str(e)}")




# ----------------------------------------------------------------
# Merchant Audio API
# ----------------------------------------------------------------
@app.post("/start-recording")
async def start_recording(current_user: str = Depends(get_current_user),):
    """ API to start recording audio. """
    if recording_event.is_set():
        raise HTTPException(status_code=400, detail="Recording is already in progress.")
    
    thread = threading.Thread(target=record_audio)
    thread.start()
    return {"message": "Recording started."}


@app.post("/stop-recording")
async def stop_recording(current_user: str = Depends(get_current_user),):
    """ API to stop recording audio and save the file. """
    if not recording_event.is_set():
        raise HTTPException(status_code=400, detail="No active recording.")

    recording_event.clear()  # Stop recording
    return {"message": "Recording stopped."}



# ----------------------------------------------------------------
# Merchant Chat API
# ----------------------------------------------------------------
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
    """ FastAPI endpoint to handle merchant queries. """
    try:
        result = query_db_for_merchant(query=request.query, audio_query=request.audio_query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
