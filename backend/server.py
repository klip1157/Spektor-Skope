from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# Detection Event Model
class DetectionEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detection_type: str  # "pose", "anomaly", "spirit_box"
    confidence: float
    keypoints_count: int = 0
    emf_level: float = 0.0
    spirit_box_frequency: Optional[float] = None
    screenshot_data: Optional[str] = None  # Base64 encoded image
    notes: Optional[str] = None

class DetectionEventCreate(BaseModel):
    detection_type: str
    confidence: float
    keypoints_count: int = 0
    emf_level: float = 0.0
    spirit_box_frequency: Optional[float] = None
    screenshot_data: Optional[str] = None
    notes: Optional[str] = None

class SessionLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_end: Optional[datetime] = None
    total_detections: int = 0
    max_emf_level: float = 0.0
    detection_events: List[str] = []  # List of detection event IDs

class SessionLogCreate(BaseModel):
    pass

class SessionLogUpdate(BaseModel):
    session_end: Optional[datetime] = None
    total_detections: Optional[int] = None
    max_emf_level: Optional[float] = None
    detection_events: Optional[List[str]] = None


# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "GhostTube SLS Camera API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


# Detection Events API
@api_router.post("/detections", response_model=DetectionEvent)
async def create_detection_event(input: DetectionEventCreate):
    event_dict = input.model_dump()
    event_obj = DetectionEvent(**event_dict)
    
    doc = event_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    await db.detection_events.insert_one(doc)
    return event_obj

@api_router.get("/detections", response_model=List[DetectionEvent])
async def get_detection_events(limit: int = 50):
    events = await db.detection_events.find(
        {}, 
        {"_id": 0}
    ).sort("timestamp", -1).to_list(limit)
    
    for event in events:
        if isinstance(event['timestamp'], str):
            event['timestamp'] = datetime.fromisoformat(event['timestamp'])
    
    return events

@api_router.get("/detections/{detection_id}", response_model=DetectionEvent)
async def get_detection_event(detection_id: str):
    event = await db.detection_events.find_one({"id": detection_id}, {"_id": 0})
    if event:
        if isinstance(event['timestamp'], str):
            event['timestamp'] = datetime.fromisoformat(event['timestamp'])
        return event
    return {"error": "Detection not found"}

@api_router.delete("/detections/{detection_id}")
async def delete_detection_event(detection_id: str):
    result = await db.detection_events.delete_one({"id": detection_id})
    if result.deleted_count:
        return {"message": "Detection deleted successfully"}
    return {"error": "Detection not found"}


# Session Logs API
@api_router.post("/sessions", response_model=SessionLog)
async def create_session():
    session_obj = SessionLog()
    
    doc = session_obj.model_dump()
    doc['session_start'] = doc['session_start'].isoformat()
    if doc['session_end']:
        doc['session_end'] = doc['session_end'].isoformat()
    
    await db.session_logs.insert_one(doc)
    return session_obj

@api_router.get("/sessions", response_model=List[SessionLog])
async def get_sessions(limit: int = 20):
    sessions = await db.session_logs.find(
        {}, 
        {"_id": 0}
    ).sort("session_start", -1).to_list(limit)
    
    for session in sessions:
        if isinstance(session['session_start'], str):
            session['session_start'] = datetime.fromisoformat(session['session_start'])
        if session.get('session_end') and isinstance(session['session_end'], str):
            session['session_end'] = datetime.fromisoformat(session['session_end'])
    
    return sessions

@api_router.put("/sessions/{session_id}", response_model=SessionLog)
async def update_session(session_id: str, update: SessionLogUpdate):
    update_dict = {k: v for k, v in update.model_dump().items() if v is not None}
    
    if 'session_end' in update_dict and update_dict['session_end']:
        update_dict['session_end'] = update_dict['session_end'].isoformat()
    
    await db.session_logs.update_one(
        {"id": session_id},
        {"$set": update_dict}
    )
    
    session = await db.session_logs.find_one({"id": session_id}, {"_id": 0})
    if session:
        if isinstance(session['session_start'], str):
            session['session_start'] = datetime.fromisoformat(session['session_start'])
        if session.get('session_end') and isinstance(session['session_end'], str):
            session['session_end'] = datetime.fromisoformat(session['session_end'])
        return session
    return {"error": "Session not found"}


# Screenshot save endpoint
@api_router.post("/screenshots")
async def save_screenshot(data: dict):
    screenshot_id = str(uuid.uuid4())
    doc = {
        "id": screenshot_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_data": data.get("image_data"),
        "detection_count": data.get("detection_count", 0),
        "emf_level": data.get("emf_level", 0),
        "notes": data.get("notes", "")
    }
    await db.screenshots.insert_one(doc)
    return {"id": screenshot_id, "message": "Screenshot saved"}

@api_router.get("/screenshots")
async def get_screenshots(limit: int = 20):
    screenshots = await db.screenshots.find(
        {},
        {"_id": 0, "image_data": 0}  # Exclude large image data from list
    ).sort("timestamp", -1).to_list(limit)
    return screenshots

@api_router.get("/screenshots/{screenshot_id}")
async def get_screenshot(screenshot_id: str):
    screenshot = await db.screenshots.find_one({"id": screenshot_id}, {"_id": 0})
    if screenshot:
        return screenshot
    return {"error": "Screenshot not found"}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
