# server.py - Advanced FastAPI Backend
import os
import io
import base64
import uuid
import json
import time
import asyncio
import hashlib  # NEW: For caching
import difflib  # NEW: For pronunciation scoring
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# FASTAPI IMPORTS
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# RATE LIMITING IMPORTS
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# AI & DATA IMPORTS
from pydantic import BaseModel, Field
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import whisper
from deep_translator import GoogleTranslator
import redis.asyncio as redis
import jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from dataclasses import dataclass
from enum import Enum

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. INITIALIZATION
load_dotenv()

# Setup Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="LingoBot Pro API",
    description="Advanced AI Language Learning Platform",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Connect Rate Limiter to App
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enhanced CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Verify API Key exists
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=api_key)

# 2. DATABASE & CACHE SETUP
DATABASE_URL = "sqlite:////app/lingobot.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis for caching and real-time features
# Use the environment variable if available, otherwise default to localhost
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url, decode_responses=True)
# 3. MODELS
class ChatMode(str, Enum):
    TUTOR = "tutor"
    EXAMINER = "examiner"
    TRANSLATOR = "translator"
    DEBATE = "debate"
    ROLEPLAY = "roleplay"

class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default=UserRole.STUDENT.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON, default={})
    progress = Column(JSON, default={})

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    messages = Column(JSON)
    language = Column(String, default="en")
    mode = Column(String, default=ChatMode.TUTOR.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    analytics = Column(JSON, default={})

class LearningPath(Base):
    __tablename__ = "learning_paths"
    id = Column(String, primary_key=True)
    title = Column(String)
    description = Column(String)
    language = Column(String)
    difficulty = Column(String)
    lessons = Column(JSON)
    estimated_hours = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    email: str
    username: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str
    role: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_text: str
    language: str = "en"
    mode: ChatMode = ChatMode.TUTOR
    context: Optional[Dict] = None
    emotion: Optional[str] = None

class AudioRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    reference_text: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"

class ProgressUpdate(BaseModel):
    skill: str
    level: int
    score: float

# 4. AUTHENTICATION
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 5. AI SERVICES

class AIService:
    def __init__(self):
        # Initialize as None for lazy loading (faster server startup)
        self.whisper_model = None
        self.embedding_model = None
        self.sentiment_analyzer = None
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _get_whisper(self):
        if self.whisper_model is None:
            logger.info("Lazy loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
        return self.whisper_model

    def _get_embedding_model(self):
        if self.embedding_model is None:
            logger.info("Lazy loading Embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model

    def _get_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            logger.info("Lazy loading Sentiment Analyzer...")
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        return self.sentiment_analyzer

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        def _transcribe():
            model = self._get_whisper()
            import tempfile
            # Create temp file for Whisper processing
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                # Transcribe
                return model.transcribe(tmp.name)["text"]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _transcribe)

    async def get_embedding(self, text: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._get_embedding_model().encode(text)
        )

    def analyze_sentiment(self, text: str) -> Dict:
        analyzer = self._get_sentiment_analyzer()
        scores = analyzer.polarity_scores(text)
        return {
            "sentiment": "positive" if scores["compound"] >= 0.05 else "negative" if scores["compound"] <= -0.05 else "neutral",
            "scores": scores
        }
    
    def analyze_grammar(self, text: str) -> Dict:
        import re
        issues = []
        if len(re.findall(r'\bvery\b', text, re.IGNORECASE)) > 2:
            issues.append("Overuse of 'very' - consider stronger adjectives")
        if len(text.split()) < 3:
            issues.append("Sentence may be too short for meaningful practice")
        punctuation_errors = len(re.findall(r'\s[.,;:!?]', text))
        if punctuation_errors > 0:
            issues.append(f"Found {punctuation_errors} potential punctuation spacing issues")
            
        return {
            "issues": issues,
            "score": max(0, 100 - len(issues) * 10),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text))
        }

    # NEW: Pronunciation Scoring Algorithm
    def calculate_pronunciation_score(self, reference_text: str, spoken_text: str) -> int:
        import re
        # Helper to clean text (remove punctuation, lower case)
        def clean(t): return re.sub(r'[^\w\s]', '', t).lower().strip()
        
        ref = clean(reference_text)
        hyp = clean(spoken_text)
        
        if not ref: return 0
        
        # Calculate similarity ratio using SequenceMatcher
        matcher = difflib.SequenceMatcher(None, ref, hyp)
        return int(matcher.ratio() * 100)

ai_service = AIService()

# 6. WEBSOCKET MANAGER
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# 7. ENHANCED SYSTEM PROMPTS
def get_system_prompt(mode: ChatMode, language: str, context: Optional[Dict] = None) -> str:
    prompts = {
        ChatMode.TUTOR: f"""
        You are an expert {language} tutor with 20 years of experience.
        Your teaching style: patient, encouraging, and precise.
        
        GUIDELINES:
        1. Keep responses conversational but educational
        2. Provide subtle corrections within context
        3. Include cultural insights when relevant
        4. End with ONE follow-up question to continue conversation
        5. Use emojis occasionally for warmth 😊
        6. Break complex concepts into digestible parts
        
        User context: {context or 'No prior context'}
        
        Current focus: Improving {language} fluency through natural conversation.
        """,
        
        ChatMode.EXAMINER: f"""
        You are a formal {language} examination board certified examiner.
        Your task: Conduct a simulated IELTS speaking test.
        
        EXAMINER PROTOCOL:
        1. Ask ONE question at a time
        2. Assess: Pronunciation, Vocabulary, Grammar, Fluency, Coherence
        3. Provide numerical score (0-9 band) after each response
        4. Note specific improvements needed
        5. Maintain professional, neutral tone
        
        Assessment in progress...
        """,
        
        ChatMode.TRANSLATOR: f"""
        You are a master translator and cultural bridge between languages.
        
        TRANSLATION PROTOCOL:
        1. Provide accurate translation
        2. Explain cultural nuances
        3. Highlight idioms and their equivalents
        4. Note formal/informal registers
        5. Suggest alternative expressions
        
        Current language pair: {language} ↔ English
        Focus on natural, idiomatic expressions.
        """,
        
        ChatMode.DEBATE: f"""
        You are a debate partner for {language} practice.
        
        DEBATE RULES:
        1. Take a position on given topic
        2. Use formal debate language
        3. Present clear arguments
        4. Challenge user's points constructively
        5. Use transition phrases for coherence
        
        Goal: Improve argumentation and persuasion skills in {language}.
        """,
        
        ChatMode.ROLEPLAY: f"""
        You are a role-play partner for {language} immersion.
        
        SCENARIO: {context.get('scenario', 'Casual conversation in a café')}
        
        ROLE-PLAY GUIDELINES:
        1. Stay in character
        2. Use appropriate register (formal/informal)
        3. Introduce natural dialogue elements
        4. Create opportunities for specific vocabulary use
        5. Make it fun and engaging!
        """
    }
    
    return prompts.get(mode, prompts[ChatMode.TUTOR])

# 8. API ENDPOINTS

@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        db_user = User(
            email=user.email,
            username=user.username,
            hashed_password=hashed_password,
            settings={
                "default_language": "en",
                "notifications": True,
                "theme": "light"
            },
            progress={
                "level": 1,
                "xp": 0,
                "streak": 0,
                "skills": {}
            }
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Create token
        access_token = create_access_token(
            data={"sub": db_user.id, "role": db_user.role}
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=db_user.id,
            username=db_user.username,
            role=db_user.role
        )
    finally:
        db.close()

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    db = SessionLocal()
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        access_token = create_access_token(
            data={"sub": db_user.id, "role": db_user.role}
        )
        
        # Update last login
        db_user.progress["last_login"] = datetime.utcnow().isoformat()
        db.commit()
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=db_user.id,
            username=db_user.username,
            role=db_user.role
        )
    finally:
        db.close()

# --- CACHE HELPER FUNCTIONS ---
async def get_cached_audio(text: str, lang: str):
    """Retrieve audio from Redis cache if available"""
    try:
        clean_text = text.strip().lower()
        key = f"tts:{lang}:{hashlib.md5(clean_text.encode()).hexdigest()}"
        cached_audio = await redis_client.get(key)
        if cached_audio:
            logger.info("⚡ Audio cache hit!")
            return cached_audio
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    return None

async def cache_audio(text: str, lang: str, audio_base64: str):
    """Save generated audio to Redis cache"""
    try:
        clean_text = text.strip().lower()
        key = f"tts:{lang}:{hashlib.md5(clean_text.encode()).hexdigest()}"
        # Expire in 24 hours (86400 seconds)
        await redis_client.setex(key, 86400, audio_base64)
    except Exception as e:
        logger.error(f"Cache set error: {e}")

# --- UPDATED CHAT ENDPOINT (Rate Limited + Cached) ---
@app.post("/chat/advanced", response_model=Dict)
@limiter.limit("15/minute")  # Limit: 15 requests per minute
async def advanced_chat(request: Request, chat_data: ChatRequest, background_tasks: BackgroundTasks):
    """Enhanced chat endpoint with analytics, caching, and rate limiting"""
    # NOTE: 'request' is now the Connection (for SlowAPI)
    # NOTE: 'chat_data' is now the User Input
    
    session_id = chat_data.session_id or str(uuid.uuid4())
    
    try:
        # 1. Analyze user input
        sentiment = ai_service.analyze_sentiment(chat_data.user_text)
        grammar = ai_service.analyze_grammar(chat_data.user_text)
        embedding = await ai_service.get_embedding(chat_data.user_text)
        
        # 2. Prepare context
        enhanced_context = {
            "sentiment": sentiment,
            "grammar_analysis": grammar,
            "user_embedding": embedding.tolist(),
            "timestamp": datetime.utcnow().isoformat(),
            **(chat_data.context or {})
        }
        
        # 3. Generate AI response
        system_prompt = get_system_prompt(chat_data.mode, chat_data.language, enhanced_context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_data.user_text}
        ]
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_reply = response.choices[0].message.content
        learning_points = extract_learning_points(ai_reply, chat_data.user_text)
        
        # 4. Generate TTS (With Smart Caching)
        
        # A. Check Cache First
        audio_base64 = await get_cached_audio(ai_reply, chat_data.language)
        
        # B. If not in cache, generate it
        if not audio_base64:
            def generate_audio_memory():
                mp3_fp = io.BytesIO()
                tts = gTTS(text=ai_reply, lang=chat_data.language, slow=False)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                return base64.b64encode(mp3_fp.read()).decode('utf-8')

            loop = asyncio.get_event_loop()
            audio_base64 = await loop.run_in_executor(None, generate_audio_memory)
            
            # C. Save to cache in background
            background_tasks.add_task(cache_audio, ai_reply, chat_data.language, audio_base64)
        
        # 5. Prepare Response
        response_data = {
            "session_id": session_id,
            "ai_reply": ai_reply,
            "learning_points": learning_points,
            "sentiment_feedback": sentiment,
            "grammar_feedback": grammar,
            "audio_base64": audio_base64,
            "enhanced_context": enhanced_context,
            "suggested_next_topics": suggest_topics(chat_data.user_text),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 6. Background tasks
        # We pass the original 'chat_data' object here so the helper function works
        background_tasks.add_task(store_conversation, session_id, chat_data, response_data)
        background_tasks.add_task(update_user_progress, session_id, chat_data, response_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- UPDATED AUDIO ENDPOINT (With Pronunciation Scoring) ---
@app.post("/audio/transcribe")
async def transcribe_audio(request: AudioRequest):
    """Transcribe audio to text with pronunciation scoring"""
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Transcribe
        transcription = await ai_service.transcribe_audio(audio_bytes)
        
        # Analyze Sentiment & Grammar
        sentiment = ai_service.analyze_sentiment(transcription)
        grammar = ai_service.analyze_grammar(transcription)
        
        # NEW: Calculate Pronunciation Score
        pronunciation_score = 0
        if request.reference_text:
            pronunciation_score = ai_service.calculate_pronunciation_score(
                request.reference_text, 
                transcription
            )
        
        return {
            "text": transcription,
            "language": request.language,
            "sentiment": sentiment,
            "grammar_score": grammar["score"],
            "pronunciation_score": pronunciation_score,
            "confidence": 0.95
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/advanced")
async def advanced_translate(request: TranslationRequest):
    """Enhanced translation with cultural context"""
    try:
        # Translate
        translated = GoogleTranslator(
            source=request.source_lang,
            target=request.target_lang
        ).translate(request.text)
        
        # Get cultural notes
        cultural_notes = await get_cultural_notes(request.text, request.target_lang)
        
        # Get alternative translations
        alternatives = await get_translation_alternatives(request.text, request.target_lang)
        
        return {
            
            "original": request.text,
            "translated": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "cultural_notes": cultural_notes,
            "alternatives": alternatives,
            "difficulty_level": assess_translation_difficulty(request.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time features"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "chat":
                # Real-time chat processing
                response = await process_realtime_chat(data["message"])
                await manager.send_personal_message(
                    json.dumps({"type": "chat_response", "data": response}),
                    client_id
                )
                
            elif message_type == "typing":
                # Typing indicator
                await manager.broadcast(
                    json.dumps({"type": "typing", "user_id": client_id})
                )
                
            elif message_type == "audio_stream":
                # Process audio stream
                await process_audio_stream(data["audio_chunk"], client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(json.dumps({"type": "user_left", "user_id": client_id}))

@app.get("/learning/path/{language}/{level}")
async def get_learning_path(language: str, level: str):
    """Get personalized learning path"""
    paths = {
        "beginner": {
            "title": f"{language.title()} Beginner Journey",
            "description": "Master the basics and build confidence",
            "weeks": 8,
            "daily_minutes": 30,
            "milestones": [
                {"week": 1, "focus": "Greetings & Introductions"},
                {"week": 2, "focus": "Common Phrases"},
                {"week": 3, "focus": "Present Tense"},
                {"week": 4, "focus": "Daily Routines"},
                {"week": 5, "focus": "Food & Ordering"},
                {"week": 6, "focus": "Directions & Transportation"},
                {"week": 7, "focus": "Shopping & Numbers"},
                {"week": 8, "focus": "Review & Cultural Insights"}
            ]
        }
    }
    return paths.get(level, paths["beginner"])

@app.post("/progress/update")
async def update_progress(update: ProgressUpdate, user_id: str = "default"):
    """Update user learning progress"""
    # Store in Redis for real-time updates
    await redis_client.hset(
        f"user:{user_id}:progress",
        update.skill,
        json.dumps({"level": update.level, "score": update.score, "timestamp": datetime.utcnow().isoformat()})
    )
    
    # Calculate overall progress
    progress_data = await redis_client.hgetall(f"user:{user_id}:progress")
    total_score = sum(json.loads(v)["score"] for v in progress_data.values())
    average_score = total_score / len(progress_data) if progress_data else 0
    
    return {
        "success": True,
        "average_score": average_score,
        "skill_count": len(progress_data),
        "updated_at": datetime.utcnow().isoformat()
    }

@app.get("/analytics/dashboard/{user_id}")
async def get_dashboard(user_id: str):
    """Get comprehensive learning analytics"""
    # Get data from Redis and database
    progress = await redis_client.hgetall(f"user:{user_id}:progress")
    
    # Calculate statistics
    total_sessions = await redis_client.get(f"user:{user_id}:session_count") or 0
    streak = await redis_client.get(f"user:{user_id}:streak") or 0
    
    # Generate insights
    insights = await generate_learning_insights(user_id)
    
    return {
        "user_id": user_id,
        "progress_summary": {
            "total_sessions": int(total_sessions),
            "current_streak": int(streak),
            "skills_learned": len(progress),
            "average_score": calculate_average_score(progress)
        },
        "weekly_activity": await get_weekly_activity(user_id),
        "skill_distribution": progress,
        "insights": insights,
        "recommendations": await get_recommendations(user_id)
    }

# 9. HELPER FUNCTIONS

async def store_conversation(session_id: str, request: ChatRequest, response: Dict):
    """Store conversation in database"""
    db = SessionLocal()
    try:
        conv = Conversation(
            session_id=session_id,
            messages=[
                {"role": "user", "content": request.user_text},
                {"role": "assistant", "content": response["ai_reply"]}
            ],
            language=request.language,
            mode=request.mode.value,
            analytics=response
        )
        db.add(conv)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to store conversation: {e}")
    finally:
        db.close()

async def update_user_progress(session_id: str, request: ChatRequest, response: Dict):
    """Update user progress based on interaction"""
    # Increment session count
    await redis_client.incr(f"user:{session_id}:session_count")
    
    # Update streak (simplified)
    today = datetime.utcnow().date().isoformat()
    last_active = await redis_client.get(f"user:{session_id}:last_active")
    
    if last_active != today:
        current_streak = int(await redis_client.get(f"user:{session_id}:streak") or 0)
        if last_active and (datetime.fromisoformat(last_active).date() == 
                           datetime.utcnow().date() - timedelta(days=1)):
            current_streak += 1
        else:
            current_streak = 1
        
        await redis_client.set(f"user:{session_id}:streak", current_streak)
        await redis_client.set(f"user:{session_id}:last_active", today)
    
    # Update XP
    xp_gained = calculate_xp(response)
    await redis_client.incrby(f"user:{session_id}:xp", xp_gained)

def extract_learning_points(ai_reply: str, user_input: str) -> List[Dict]:
    """Extract specific learning points from conversation"""
    points = []
    
    # Vocabulary extraction
    import re
    vocabulary_patterns = [
        r'\b(\w+)\b.*?(?:means|refers to|is called)',
        r'Key (?:word|phrase):?\s*(\w+(?:\s+\w+)*)',
        r'Remember.*?\b(\w+(?:\s+\w+)*)\b'
    ]
    
    for pattern in vocabulary_patterns:
        matches = re.findall(pattern, ai_reply, re.IGNORECASE)
        for match in matches:
            points.append({
                "type": "vocabulary",
                "content": match,
                "context": "From conversation",
                "example": find_example_in_reply(ai_reply, match)
            })
    
    # Grammar points
    grammar_keywords = ["grammar", "tense", "plural", "singular", "conjugation"]
    for keyword in grammar_keywords:
        if keyword in ai_reply.lower():
            points.append({
                "type": "grammar",
                "topic": keyword.title(),
                "explanation": extract_paragraph_containing(ai_reply, keyword)
            })
    
    return points

def suggest_topics(user_input: str) -> List[str]:
    """Suggest next conversation topics based on current input"""
    topics = {
        "food": ["restaurant ordering", "cooking vocabulary", "dietary restrictions"],
        "travel": ["directions", "accommodation", "transportation"],
        "work": ["job interviews", "office communication", "professional email"],
        "daily": ["morning routine", "shopping", "social plans"],
        "hobbies": ["sports", "music", "reading", "movies"]
    }
    
    input_lower = user_input.lower()
    suggested = []
    
    for category, category_topics in topics.items():
        if any(word in input_lower for word in category.split()):
            suggested.extend(category_topics[:2])
    
    return suggested[:3] if suggested else ["introductions", "weather", "future plans"]

async def get_cultural_notes(text: str, target_lang: str) -> List[str]:
    """Get cultural context for translation"""
    notes = []
    
    # This would integrate with a cultural database or API
    cultural_data = {
        "en": {
            "thanks": "In English-speaking cultures, 'thank you' is used very frequently",
            "please": "Using 'please' makes requests much more polite",
            "sorry": "English speakers apologize often, even for minor inconveniences"
        },
        "es": {
            "gracias": "In Spanish culture, expressing gratitude is important",
            "por favor": "Always use 'por favor' when making requests",
            "perdón": "Used similarly to English 'sorry'"
        }
        # Add more languages
    }
    
    # Simple keyword matching
    for keyword in text.split()[:5]:  # Check first few words
        if target_lang in cultural_data and keyword.lower() in cultural_data[target_lang]:
            notes.append(cultural_data[target_lang][keyword.lower()])
    
    return notes if notes else ["No specific cultural notes for this text"]

def calculate_xp(response: Dict) -> int:
    """Calculate XP gained from interaction"""
    base_xp = 10
    grammar_bonus = response.get("grammar_feedback", {}).get("score", 0) / 10
    sentiment_bonus = 5 if response.get("sentiment_feedback", {}).get("sentiment") == "positive" else 0
    length_bonus = min(len(response.get("ai_reply", "")) // 50, 10)
    
    return int(base_xp + grammar_bonus + sentiment_bonus + length_bonus)

async def generate_learning_insights(user_id: str) -> List[str]:
    """Generate personalized learning insights"""
    insights = []
    
    # Get user data
    progress = await redis_client.hgetall(f"user:{user_id}:progress")
    conversations = await get_recent_conversations(user_id)
    
    if not progress:
        insights.append("Welcome! Start by practicing greetings and introductions.")
        return insights
    
    # Generate insights based on data
    if len(progress) < 3:
        insights.append("You're just starting! Try different conversation topics to build vocabulary.")
    
    avg_score = calculate_average_score(progress)
    if avg_score < 70:
        insights.append("Focus on grammar exercises to improve your accuracy.")
    elif avg_score > 85:
        insights.append("Great progress! Try more challenging topics like debates or role-plays.")
    
    # Time-based insights
    last_active = await redis_client.get(f"user:{user_id}:last_active")
    if last_active:
        last_date = datetime.fromisoformat(last_active).date()
        days_inactive = (datetime.utcnow().date() - last_date).days
        
        if days_inactive > 2:
            insights.append(f"You haven't practiced in {days_inactive} days. Regular practice is key!")
        elif days_inactive == 0:
            insights.append("You practiced today! Consistency will lead to fluency.")
    
    return insights[:3]  # Return top 3 insights

# 10. UTILITY FUNCTIONS

def extract_paragraph_containing(text: str, keyword: str) -> str:
    """Extract paragraph containing a keyword"""
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        if keyword.lower() in para.lower():
            return para.strip()
    return ""

def find_example_in_reply(text: str, word: str) -> str:
    """Find example usage of a word in text"""
    sentences = text.split('. ')
    for sentence in sentences:
        if word.lower() in sentence.lower():
            return sentence.strip()
    return ""

def assess_translation_difficulty(text: str) -> str:
    """Assess difficulty level of translation"""
    word_count = len(text.split())
    if word_count <= 5:
        return "easy"
    elif word_count <= 15:
        return "medium"
    else:
        return "hard"

def calculate_average_score(progress_data: Dict) -> float:
    """Calculate average score from progress data"""
    if not progress_data:
        return 0.0
    
    scores = [json.loads(v)["score"] for v in progress_data.values()]
    return sum(scores) / len(scores)

async def get_recent_conversations(user_id: str, limit: int = 10):
    """Get recent conversations from database"""
    db = SessionLocal()
    try:
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(
            Conversation.created_at.desc()
        ).limit(limit).all()
        
        return [conv.messages for conv in conversations]
    finally:
        db.close()

async def get_weekly_activity(user_id: str):
    """Get weekly activity data"""
    # This would query the database for activity by day
    return {
        "monday": 3,
        "tuesday": 5,
        "wednesday": 2,
        "thursday": 4,
        "friday": 6,
        "saturday": 1,
        "sunday": 0
    }

async def get_recommendations(user_id: str):
    """Get personalized learning recommendations"""
    return [
        "Practice past tense conversations",
        "Try the 'Restaurant Role-play' exercise",
        "Review vocabulary from last week",
        "Challenge yourself with the 'Debate Mode'"
    ]

async def process_realtime_chat(message: str) -> Dict:
    """Process chat message in real-time"""
    # Quick response without full processing
    quick_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": message}],
        temperature=0.7,
        max_tokens=100
    )
    
    return {
        "text": quick_response.choices[0].message.content,
        "is_typing": False,
        "timestamp": datetime.utcnow().isoformat()
    }

async def process_audio_stream(audio_chunk: str, client_id: str):
    """Process streaming audio"""
    # This would handle real-time audio processing
    pass

# 11. STATIC FILES (For web dashboard)
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

# 12. STARTUP AND SHUTDOWN
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting LingoBot Pro API...")
    
    # Test Redis connection
    try:
        await redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    # Load ML models (async)
    # await ai_service.load_models()
    
    logger.info("LingoBot Pro API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down LingoBot Pro API...")
    await redis_client.close()
    logger.info("Cleanup complete")

# 13. MAIN ENTRY POINT
if __name__ == "__main__":
    import uvicorn
    
    # Enhanced configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "workers": 4,
        "log_level": "info",
        "access_log": True
    }
    
    logger.info(f"Starting server on {config['host']}:{config['port']}")
    uvicorn.run("server:app", **config)