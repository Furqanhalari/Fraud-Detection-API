"""
GET /dashboard — Serve the Jinja2 monitoring dashboard.
"""

import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "templates")

router = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")
