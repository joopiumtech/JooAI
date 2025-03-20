import os
import ast

from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from utils import initialize_db, verify_password

from dotenv import load_dotenv
load_dotenv(override=True)

# Secret Key for JWT
SECRET_KEY = os.environ.get("JWT_SECRET")
ALGORITHM = os.environ.get("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth")

# Hashing password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(data: dict):
    to_encode = data.copy()
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)




def auth_test(email: str, password: str):
    try:
        db = initialize_db()
        
        email_query = f"""SELECT COUNT(*) > 0 AS user_exists FROM users WHERE email = '{email}';"""
        email_address = ast.literal_eval(db.run(email_query))

        if email_address[0][0] == 0:
            return {"is_authenticated": False,
                    "auth_message": "Email address not found. Please check your email address and try again."} 
        else:
            password_get_query = f"""SELECT password FROM users where email='{email}'"""
            hashed_password = ast.literal_eval(db.run(password_get_query))
            is_password_correct = verify_password(password, hashed_password[0][0])


            if is_password_correct:
                # Generate JWT token
                access_token = create_access_token(data={"sub": email})
                return {"is_authenticated": True,
                        "auth_message": "Authenticated successfully",
                        "access_token": access_token}
            else:
                return {"is_authenticated": False,
                        "auth_message": "Wrong password. Please check your password and try again."}
    except Exception as error:
        return {
            "is_authenticated": False,
            "auth_message": f"Oops! Something went wrong. Please try again."
        }
    



def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validates JWT token and returns user email.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")

        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )