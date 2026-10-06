from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.security import hash_password, verify_password, create_access_token
from app.repositories import user_repo
from app.schemas.user import UserCreate
from app.schemas.auth import LoginRequest, TokenResponse


def register_user(db: Session, payload: UserCreate) -> TokenResponse:
    if payload.password != payload.confirm_password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Passwords do not match.")
    if user_repo.get_by_email(db, payload.email):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    user = user_repo.create_user(
        db, name=payload.name, email=payload.email,
        password_hash=hash_password(payload.password), role="user",
    )
    token = create_access_token(subject=user.id, role=user.role.value)
    return TokenResponse(access_token=token, role=user.role.value, user_id=user.id, name=user.name)


def login_user(db: Session, payload: LoginRequest) -> TokenResponse:
    user = user_repo.get_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    token = create_access_token(subject=user.id, role=user.role.value)
    return TokenResponse(access_token=token, role=user.role.value, user_id=user.id, name=user.name)
