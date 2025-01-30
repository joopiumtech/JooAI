import os
from datetime import date, datetime, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from merchant_backend import query_db_for_merchant
from user_backend import book_table, query_db_for_user

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()


# ----------------------------------------------------------------
# Merchant Query API
# ----------------------------------------------------------------

# Request model for merchant query
class MerchantQueryRequest(BaseModel):
    email: str
    query: str


# Response model for merchant query
class MerchantQueryResponse(BaseModel):
    ai_response: str



@app.post("/merchant/chat", response_model=MerchantQueryResponse)
async def merchant_query(request: MerchantQueryRequest):
    """
    FastAPI endpoint to handle merchant queries.

    Args:
        request (MerchantQueryRequest): The request body containing the merchant query.

    Returns:
        MerchantQueryResponse: The response containing the query and AI-generated response.

    Raises:
        HTTPException: If an error occurs during query processing.
    """
    try:
        result = query_db_for_merchant(email=request.email, query=request.query)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



# ----------------------------------------------------------------
# User Query API
# ----------------------------------------------------------------

# Request model for user query
class UserQueryRequest(BaseModel):
    email: str
    query: str


# Response model for user query
class UserQueryResponse(BaseModel):
    ai_response: str


@app.post("/user/chat", response_model=UserQueryResponse)
async def user_query(request: UserQueryRequest):
    try:
        result = query_db_for_user(email=request.email, query=request.query)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ----------------------------------------------------------------
# User Bookings API
# ----------------------------------------------------------------

# Request model for table booking
class BookTableRequest(BaseModel):
    name: str
    phone: str
    email: str
    date: date
    time: time
    guests: str
    message: str
    booking_type: str


# Response model for table booking
class BookTableResponse(BaseModel):
    ai_response: str
    name: str
    phone: str
    email: str
    date: date
    time: time
    guests: str
    message: str
    booking_type: str


@app.post("/user/book_table", response_model=BookTableResponse)
async def user_booking(request: BookTableRequest):
    try:
        # Generate timestamps
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Call the helper function
        result = book_table(
            name=request.name,
            phone=request.phone,
            email=request.email,
            date=request.date,
            time=request.time,
            guests=request.guests,
            message=request.message,
            booking_type=request.booking_type,
            created_at=created_at,
            updated_at=updated_at,
        )
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
