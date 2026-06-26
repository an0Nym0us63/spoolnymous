from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from ....db.session import get_db
from ....core.security import verify_password, create_access_token, decode_token, hash_password
from ....services.settings_service import get_setting, set_setting

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    stored_hash = await get_setting(db, "ADMIN_PASSWORD_HASH")
    stored_user = await get_setting(db, "ADMIN_USERNAME", "admin")

    # Premier démarrage : pas de hash → mot de passe par défaut "admin"
    if not stored_hash:
        if form.username != "admin" or form.password != "admin":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
    else:
        if form.username != stored_user or not verify_password(form.password, stored_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")

    token = create_access_token(sub=form.username)
    return {"access_token": token}


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> str:
    username = decode_token(token)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
    return username
