from fastapi import APIRouter
from .routes import auth, printer, settings

router = APIRouter(prefix="/api/v1")
router.include_router(auth.router,    prefix="/auth",     tags=["auth"])
router.include_router(printer.router, prefix="/printer",  tags=["printer"])
router.include_router(settings.router,prefix="/settings", tags=["settings"])
