from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, APIRouter, HTTPException, Request, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
import os
import logging
import bcrypt
import jwt
import secrets
import re
import base64
import io
import json
import numpy as np
import faiss
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.environ.get("MEMORY_DIR", os.path.join(PROJECT_ROOT, "memory"))
CERTIFICATE_ASSET_DIR = os.environ.get(
    "CERTIFICATE_ASSET_DIR",
    os.path.join(PROJECT_ROOT, "frontend", "public", "certificates"),
)
CERTIFICATE_PUBLIC_PATH = "/certificates"
RESUME_ASSET_DIR = os.environ.get(
    "RESUME_ASSET_DIR",
    os.path.join(PROJECT_ROOT, "frontend", "public", "resume"),
)
RESUME_PUBLIC_PATH = "/resume"
DEFAULT_RESUME_FILENAME = "varun-resume.pdf"
ALLOWED_CERTIFICATE_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

class MemoryInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class MemoryUpdateResult:
    def __init__(self, matched_count=0, modified_count=0, upserted_id=None):
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.upserted_id = upserted_id


class MemoryDeleteResult:
    def __init__(self, deleted_count=0):
        self.deleted_count = deleted_count


class MemoryCursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def sort(self, sort_spec, direction=None):
        if isinstance(sort_spec, list):
            specs = sort_spec
        else:
            specs = [(sort_spec, direction or 1)]

        for key, order in reversed(specs):
            self.docs.sort(key=lambda doc: (doc.get(key) is None, doc.get(key)), reverse=order == -1)
        return self

    def limit(self, count):
        self.docs = self.docs[:count]
        return self

    async def to_list(self, length):
        return [dict(doc) for doc in self.docs[:length]]


class MemoryCollection:
    def __init__(self):
        self.docs = []

    async def create_index(self, *args, **kwargs):
        return None

    def _matches(self, doc, query):
        if not query:
            return True

        for key, value in query.items():
            if key == "$or":
                if not any(self._matches(doc, clause) for clause in value):
                    return False
                continue

            doc_value = doc.get(key)
            if isinstance(value, dict):
                if "$gte" in value and doc_value < value["$gte"]:
                    return False
                continue

            if doc_value != value:
                return False

        return True

    def _apply_update(self, doc, update):
        for key, value in update.get("$set", {}).items():
            doc[key] = value
        for key, value in update.get("$setOnInsert", {}).items():
            doc.setdefault(key, value)

    async def find_one(self, query, projection=None):
        for doc in self.docs:
            if self._matches(doc, query):
                return dict(doc)
        return None

    async def insert_one(self, doc):
        stored = dict(doc)
        stored.setdefault("_id", ObjectId())
        self.docs.append(stored)
        return MemoryInsertResult(stored["_id"])

    async def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if self._matches(doc, query):
                self._apply_update(doc, update)
                return MemoryUpdateResult(matched_count=1, modified_count=1)

        if upsert:
            doc = dict(query)
            self._apply_update(doc, update)
            doc.setdefault("_id", doc.get("_id", ObjectId()))
            self.docs.append(doc)
            return MemoryUpdateResult(upserted_id=doc["_id"])

        return MemoryUpdateResult()

    async def delete_one(self, query):
        for index, doc in enumerate(self.docs):
            if self._matches(doc, query):
                self.docs.pop(index)
                return MemoryDeleteResult(1)
        return MemoryDeleteResult()

    async def count_documents(self, query):
        return sum(1 for doc in self.docs if self._matches(doc, query))

    def find(self, query=None, projection=None):
        docs = []
        for doc in self.docs:
            if self._matches(doc, query or {}):
                result = dict(doc)
                if projection:
                    for key, include in projection.items():
                        if include == 0:
                            result.pop(key, None)
                docs.append(result)
        return MemoryCursor(docs)

    def aggregate(self, pipeline):
        docs = [dict(doc) for doc in self.docs]
        for stage in pipeline:
            if "$match" in stage:
                docs = [doc for doc in docs if self._matches(doc, stage["$match"])]
            elif "$group" in stage:
                group_key = stage["$group"]["_id"].lstrip("$")
                grouped = {}
                for doc in docs:
                    key = doc.get(group_key)
                    grouped.setdefault(key, {"_id": key, "count": 0})
                    grouped[key]["count"] += 1
                docs = list(grouped.values())
            elif "$sort" in stage:
                for key, order in reversed(list(stage["$sort"].items())):
                    docs.sort(key=lambda doc: doc.get(key) or 0, reverse=order == -1)
            elif "$limit" in stage:
                docs = docs[:stage["$limit"]]
        return MemoryCursor(docs)


class MemoryAsyncDatabase:
    def __init__(self):
        self._collections = {}

    def __getattr__(self, name):
        if name not in self._collections:
            self._collections[name] = MemoryCollection()
        return self._collections[name]


def create_memory_database():
    logger.warning("MongoDB is unavailable; using in-memory development database.")
    return MemoryAsyncDatabase()


# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'default-secret-change-me')
JWT_ALGORITHM = "HS256"

# Create the main app
app = FastAPI(title="Varun's AI Portfolio API")

# Create router with /api prefix
api_router = APIRouter(prefix="/api")

# ============= PYDANTIC MODELS =============

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    created_at: str

class ProjectCreate(BaseModel):
    title: str
    description: str
    technologies: List[str]
    image_url: Optional[str] = None
    github_url: Optional[str] = None
    demo_url: Optional[str] = None
    category: str = "data-science"
    featured: bool = False
    year: Optional[str] = None

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = None
    image_url: Optional[str] = None
    github_url: Optional[str] = None
    demo_url: Optional[str] = None
    category: Optional[str] = None
    featured: Optional[bool] = None
    year: Optional[str] = None

class ProjectRestore(BaseModel):
    id: str
    title: str
    description: str
    technologies: List[str]
    image_url: Optional[str] = None
    github_url: Optional[str] = None
    demo_url: Optional[str] = None
    category: str = "data-science"
    featured: bool = False
    year: Optional[str] = None
    source_key: Optional[str] = None
    sort_order: Optional[int] = None
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    updated_at: Optional[str] = None

class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class KnowledgeBaseEntry(BaseModel):
    category: str
    content: str
    metadata: Optional[dict] = None

class FaceRegister(BaseModel):
    face_data: str  # Base64 encoded image
    user_id: Optional[str] = None

class AnalyticsEvent(BaseModel):
    event_type: str
    page: Optional[str] = None
    metadata: Optional[dict] = None

class EducationEntry(BaseModel):
    degree: str
    institution: str
    location: str
    period: str
    coursework: Optional[List[str]] = None

class CertificationEntry(BaseModel):
    name: str
    institution: str
    period: str
    credential_url: Optional[str] = None

class PortfolioInfoUpdate(BaseModel):
    skills: Optional[Dict[str, List[str]]] = None
    education: Optional[List[EducationEntry]] = None
    certifications: Optional[List[CertificationEntry]] = None

class ResumeInfo(BaseModel):
    title: str = ""
    filename: str = DEFAULT_RESUME_FILENAME
    url: str = f"{RESUME_PUBLIC_PATH}/{DEFAULT_RESUME_FILENAME}"
    content_type: str = "application/pdf"
    size: int = 0
    uploaded_at: Optional[str] = None
    uploaded_by: Optional[str] = None

class HomeStatEntry(BaseModel):
    label: str
    value: int = Field(ge=0)
    suffix: str = "+"

class HomeSkillEntry(BaseModel):
    name: str
    level: int = Field(ge=0, le=100)

class HomeInfoUpdate(BaseModel):
    stats: Optional[List[HomeStatEntry]] = None
    skills: Optional[List[HomeSkillEntry]] = None

class ProfileInfoUpdate(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    field_of_study: Optional[str] = None
    tagline: Optional[str] = None
    bio: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    social: Optional[Dict[str, str]] = None
    profile_image_url: Optional[str] = None

DEFAULT_PORTFOLIO_INFO = {
    "name": "Varun",
    "title": "Data Scientist & ML Engineer",
    "field_of_study": "Computer Science",
    "tagline": "Transforming Data into Actionable Insights",
    "bio": "B.Tech Computer Science student at Sardar Beant Singh State University with a passion for Data Science and Machine Learning. Proficient in Python, SQL, and statistical analysis with hands-on experience in building recommendation systems and data analysis pipelines.",
    "location": "Pathankot, Punjab, India",
    "email": "varunsharma1234566@gmail.com",
    "phone": "+91 6239753187",
    "profile_image_url": "https://static.prod-images.emergentagent.com/jobs/dd8d4152-3726-4bcf-afd9-da6518a514c1/images/f23f4b73d684e0d789993f4e15e4e9e7654806dc57bf09052fb0ebf6c99f92cd.png",
    "social": {
        "linkedin": "https://www.linkedin.com/in/varun-sharma-4525b1343",
        "github": "https://github.com/varun72004",
        "instagram": "https://www.instagram.com/_ordinary_boy14/",
    },
    "education": [
        {
            "degree": "Bachelor of Technology in Computer Science",
            "institution": "Sardar Beant Singh State University",
            "location": "Gurdaspur, Punjab",
            "period": "June 2022 - June 2026",
            "coursework": ["Data Structures", "Algorithms", "DBMS", "Machine Learning", "Statistics"],
        },
        {
            "degree": "Advanced Training in AI and Data Science",
            "institution": "Intellipaat School of Technology",
            "location": "Remote",
            "period": "June 2025 - June 2026",
        },
        {
            "degree": "Training in Data Science",
            "institution": "Alpha IT Managed Services",
            "location": "SAS Nagar Mohali, Punjab",
            "period": "January 2026 - July 2026",
        },
        {
            "degree": "High School (Non-Medical with Computer Science)",
            "institution": "Kendriya Vidyalaya No.2 Army Area",
            "location": "Pathankot, Punjab",
            "period": "June 2010 - June 2022",
        },
    ],
    "skills": {
        "programming": ["Python", "SQL"],
        "libraries": ["Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn", "Streamlit"],
        "ml": ["Linear & Logistic Regression", "Classification Models", "Feature Engineering", "Recommendation Systems"],
        "statistics": ["Hypothesis Testing", "Regression Analysis", "Correlation", "Probability Distributions"],
        "data_analysis": ["EDA", "Data Cleaning", "Data Preprocessing", "Trend Analysis"],
        "tools": ["Git", "GitHub", "Jupyter Notebook", "Google Colab", "VS Code", "Power BI", "Microsoft SQL Server"],
    },
    "certifications": [
        {
            "name": "Microsoft SQL Certification Training",
            "institution": "Intellipaat",
            "period": "June 2025 - July 2025",
            "credential_url": "/certificates/intellipaat-microsoft-sql-certification-training.pdf",
        },
        {
            "name": "Python Certification Course",
            "institution": "Intellipaat",
            "period": "2025",
            "credential_url": "/certificates/intellipaat-python-certification-course.pdf",
        },
        {
            "name": "Data Science Certification",
            "institution": "Coder Roots, Mohali",
            "period": "June 2025 - July 2025",
            "credential_url": "/certificates/data-science-certification.pdf",
        },
        {
            "name": "Industrial Training in Python",
            "institution": "Tech World Institute, Pathankot",
            "period": "June 2024 - July 2024",
            "credential_url": "/certificates/tech-world-python-industrial-training.jpg",
        },
    ],
}

DEFAULT_HOME_INFO = {
    "stats": [
        {"label": "Projects Completed", "value": 15, "suffix": "+"},
        {"label": "Technologies", "value": 20, "suffix": "+"},
        {"label": "Lines of Code", "value": 50000, "suffix": "K+"},
        {"label": "Certifications", "value": 4, "suffix": "+"},
    ],
    "skills": [
        {"name": "Python", "level": 95},
        {"name": "Machine Learning", "level": 88},
        {"name": "Data Analysis", "level": 92},
        {"name": "SQL", "level": 85},
        {"name": "Deep Learning", "level": 78},
        {"name": "Data Visualization", "level": 90},
    ],
}

PROFILE_FIELDS = (
    "name",
    "title",
    "field_of_study",
    "tagline",
    "bio",
    "location",
    "email",
    "phone",
    "social",
    "profile_image_url",
)
PROFILE_UNDO_DOC_ID = "profile_info_undo"
MAX_PROFILE_UNDO_HISTORY = 25

# Projects that previously lived only in the React projects page.
DEFAULT_PROJECTS = [
    {
        "source_key": "netflix-recommendation",
        "title": "Netflix Recommendation System",
        "description": "Built a collaborative filtering recommendation system that predicts user ratings based on viewing patterns. Implemented matrix factorization techniques to uncover latent factors in user-movie interactions.",
        "technologies": ["Python", "Pandas", "Scikit-learn", "Machine Learning", "NumPy"],
        "image_url": "https://static.prod-images.emergentagent.com/jobs/dd8d4152-3726-4bcf-afd9-da6518a514c1/images/d6df2e70b0d4187db8c06fb6736396a104c7fa6e16b02dc2c9cf0140c06b1c42.png",
        "github_url": "https://github.com/varun72004",
        "category": "machine-learning",
        "featured": True,
        "year": "2025",
    },
    {
        "source_key": "disease-environment",
        "title": "Disease & Environment Correlation Analysis",
        "description": "Analyzed correlations between environmental conditions (temperature, humidity, pollution) and disease outbreaks. Built an interactive Streamlit dashboard integrating real-time and historical data.",
        "technologies": ["Python", "Streamlit", "Pandas", "Data Visualization", "Statistical Analysis"],
        "image_url": "https://static.prod-images.emergentagent.com/jobs/dd8d4152-3726-4bcf-afd9-da6518a514c1/images/c800a07c3db868c1bc955ff95b204608212f754acf3988328bbda40b1da0684c.png",
        "github_url": "https://github.com/varun72004",
        "category": "data-analysis",
        "featured": True,
        "year": "2025",
    },
    {
        "source_key": "covid-analysis",
        "title": "COVID-19 Data Analysis & Forecasting",
        "description": "Comprehensive analysis of global COVID-19 trends using confirmed cases, deaths, and recovery data. Created compelling visualizations and statistical trend analysis for pandemic progression.",
        "technologies": ["Python", "Pandas", "Matplotlib", "Seaborn", "Time Series"],
        "image_url": "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=600",
        "github_url": "https://github.com/varun72004",
        "category": "data-analysis",
        "featured": True,
        "year": "2025",
    },
    {
        "source_key": "sentiment-analysis",
        "title": "Twitter Sentiment Analysis",
        "description": "Twitter Sentiment Analysis Web App using Machine Learning & NLP | Built with TF-IDF, Multinomial Naive Bayes model & Streamlit for real-time sentiment prediction.",
        "technologies": ["Python", "NLP", "Machine Learning", "Streamlit"],
        "image_url": "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=600",
        "github_url": "https://github.com/varun72004",
        "category": "deep-learning",
        "featured": False,
        "year": "2025",
    },
    {
        "source_key": "banking-management",
        "title": "Bank Management System",
        "description": "A GUI-based bank management application built with Python and Tkinter. It allows admins and customers to perform various banking operations like creating accounts, transactions, and account management through an intuitive interface.",
        "technologies": ["Python", "Tkinter (GUI)", "Financial Analysis"],
        "image_url": "https://static.vecteezy.com/system/resources/previews/010/518/833/original/digital-finance-and-banking-investment-service-on-microchip-with-cloud-computing-in-futuristic-background-bank-building-with-online-payment-secure-money-and-financial-innovation-technology-vector.jpg",
        "github_url": "https://github.com/varun72004",
        "category": "basic-python",
        "featured": False,
        "year": "2023",
    },
    {
        "source_key": "image-classifier",
        "title": "Image Classification CNN",
        "description": "Convolutional Neural Network for multi-class image classification. Achieved 96% accuracy on custom dataset using transfer learning with ResNet50.",
        "technologies": ["Python", "TensorFlow", "CNN", "Transfer Learning", "Computer Vision"],
        "image_url": "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=600",
        "github_url": "https://github.com/varun72004",
        "category": "deep-learning",
        "featured": False,
        "year": "2024",
    },
    {
        "source_key": "customer-churn-analysis",
        "title": "Customer Churn Analysis",
        "description": "This project focuses on analyzing customer churn in a telecommunications company and building predictive models to identify at-risk customers. By understanding customer behavior and churn patterns, the project aims to deliver actionable insights and recommendations that help reduce churn and improve customer retention strategies.",
        "technologies": ["Python", "NumPy", "Data Analysis", "Visualization"],
        "image_url": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=600",
        "github_url": "https://github.com/varun72004",
        "category": "data-analysis",
        "featured": False,
        "year": "2024",
    },
    {
        "source_key": "healthcare-ai",
        "title": "HealthCare AI - Diagnosis & Recommendation System",
        "description": "A comprehensive healthcare application that uses artificial intelligence to predict diseases, recommend medicines, suggest diet plans, and generate personalized daily routines based on user symptoms and health data.",
        "technologies": ["Python", "Streamlit", "Machine Learning", "Pandas", "Scikit-learn"],
        "image_url": "https://www.hepmade.com/wp-content/uploads/2025/06/ai.medicine.jpg",
        "github_url": "https://github.com/varun72004",
        "category": "machine-learning",
        "featured": False,
        "year": "2024",
    },
]

def serialize_project(project: dict) -> dict:
    response = dict(project)
    response["id"] = str(response.pop("_id"))
    return response

def parse_project_id(project_id: str) -> ObjectId:
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return ObjectId(project_id)

async def seed_default_projects():
    seed_key = "default_projects_seeded_v2"
    if await db.app_settings.find_one({"_id": seed_key}):
        return

    now = datetime.now(timezone.utc).isoformat()
    inserted_count = 0

    for sort_order, project in enumerate(DEFAULT_PROJECTS):
        existing = await db.projects.find_one({
            "$or": [
                {"source_key": project["source_key"]},
                {"title": project["title"]},
            ]
        })

        if existing:
            await db.projects.update_one(
                {"_id": existing["_id"]},
                {"$set": {"source_key": project["source_key"], "sort_order": sort_order}}
            )
            continue

        project_doc = {
            **project,
            "sort_order": sort_order,
            "created_at": now,
            "created_by": "system",
        }
        await db.projects.insert_one(project_doc)
        inserted_count += 1

    await db.app_settings.update_one(
        {"_id": seed_key},
        {"$set": {"seeded_at": now, "inserted_count": inserted_count}},
        upsert=True,
    )
    logger.info(f"Default projects seeded: {inserted_count} inserted")

def default_portfolio_info() -> dict:
    return json.loads(json.dumps(DEFAULT_PORTFOLIO_INFO))

def default_home_info() -> dict:
    return json.loads(json.dumps(DEFAULT_HOME_INFO))

async def load_portfolio_info() -> dict:
    doc = await db.app_settings.find_one({"_id": "portfolio_info"})
    if not doc:
        return default_portfolio_info()

    info = default_portfolio_info()
    info.update(doc.get("data", {}))
    return info

async def load_home_info() -> dict:
    doc = await db.app_settings.find_one({"_id": "home_info"})
    if not doc:
        return default_home_info()

    info = default_home_info()
    info.update(doc.get("data", {}))
    return info

async def load_resume_info() -> dict:
    doc = await db.app_settings.find_one({"_id": "resume_info"})
    default_path = os.path.join(RESUME_ASSET_DIR, DEFAULT_RESUME_FILENAME)
    default_size = os.path.getsize(default_path) if os.path.exists(default_path) else 0
    default_info = ResumeInfo(size=default_size).model_dump()
    if not doc:
        return default_info

    info = default_info
    info.update(doc.get("data", {}))
    return info

async def save_resume_info(resume_info: dict):
    now = datetime.now(timezone.utc).isoformat()
    await db.app_settings.update_one(
        {"_id": "resume_info"},
        {
            "$set": {
                "data": resume_info,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

async def seed_default_portfolio_info():
    existing = await db.app_settings.find_one({"_id": "portfolio_info"})
    if existing:
        return

    await db.app_settings.insert_one({
        "_id": "portfolio_info",
        "data": default_portfolio_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "system",
    })
    logger.info("Default portfolio info seeded")

async def seed_default_home_info():
    existing = await db.app_settings.find_one({"_id": "home_info"})
    if existing:
        return

    await db.app_settings.insert_one({
        "_id": "home_info",
        "data": default_home_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "system",
    })
    logger.info("Default home info seeded")

def serialize_contact_message(message: dict) -> dict:
    response = dict(message)
    response["id"] = str(response.pop("_id"))
    return response

def format_list(values: Optional[List[str]], fallback: str = "Not listed") -> str:
    values = [value for value in (values or []) if value]
    return ", ".join(values) if values else fallback

def build_site_context(portfolio_info: dict, projects: List[dict]) -> str:
    skills = portfolio_info.get("skills", {}) or {}
    skill_lines = [
        f"{category.replace('_', ' ').title()}: {format_list(skill_list)}"
        for category, skill_list in skills.items()
    ]
    education_lines = [
        f"{entry.get('degree')} at {entry.get('institution')} ({entry.get('period')})"
        for entry in portfolio_info.get("education", [])
        if entry.get("degree") and entry.get("institution")
    ]
    certification_lines = [
        f"{entry.get('name')} from {entry.get('institution')} ({entry.get('period')})"
        for entry in portfolio_info.get("certifications", [])
        if entry.get("name") and entry.get("institution")
    ]
    project_lines = [
        f"{project.get('title')}: {project.get('description')} Technologies: {format_list(project.get('technologies'))}. Year: {project.get('year') or 'Not listed'}."
        for project in projects
    ]

    social = portfolio_info.get("social", {}) or {}
    return "\n".join([
        f"Name: {portfolio_info.get('name', 'Varun')}",
        f"Title: {portfolio_info.get('title', 'Data Scientist & ML Engineer')}",
        f"Field of study: {portfolio_info.get('field_of_study', 'Computer Science')}",
        f"Tagline: {portfolio_info.get('tagline', '')}",
        f"Bio: {portfolio_info.get('bio', '')}",
        f"Location: {portfolio_info.get('location', '')}",
        f"Email: {portfolio_info.get('email', '')}",
        f"Phone: {portfolio_info.get('phone', '')}",
        f"LinkedIn: {social.get('linkedin', '')}",
        f"GitHub: {social.get('github', '')}",
        f"Instagram: {social.get('instagram', '')}",
        "Skills:\n" + "\n".join(skill_lines),
        "Education:\n" + "\n".join(education_lines),
        "Certifications:\n" + "\n".join(certification_lines),
        "Projects:\n" + "\n".join(project_lines),
    ])

def detect_chat_intent(message: str) -> str:
    query = message.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", query).strip()
    greeting_words = {"hi", "hello", "hey", "hii", "yo", "namaste", "good morning", "good afternoon", "good evening"}
    if normalized in greeting_words or re.fullmatch(r"(hi|hello|hey|hii|yo|namaste)(\s+(there|varun|bot|buddy|bro))?", normalized):
        return "greeting"

    intent_keywords = {
        "career_goals": ["goal", "future", "career", "aspire", "aim", "plan"],
        "collaboration": ["collaborate", "hire", "work together", "freelance", "opportunity", "contact", "reach"],
        "experience": ["experience", "intern", "background", "worked", "hands-on"],
        "projects": ["project", "portfolio", "built", "demo", "github"],
        "skills": ["skill", "technology", "tech", "python", "sql", "machine learning", "ml", "tools"],
        "education": ["education", "study", "college", "university", "degree", "school"],
        "certifications": ["certificate", "certification", "course", "training"],
        "resume": ["resume", "cv", "download"],
    }
    for intent, keywords in intent_keywords.items():
        if any(keyword in query for keyword in keywords):
            return intent
    return "general"

def is_low_context_message(message: str) -> bool:
    terms = re.findall(r"[a-z0-9]+", message.lower())
    return len(terms) <= 2

def build_knowledge_chunks(portfolio_info: dict, projects: List[dict], resume_info: Optional[dict] = None) -> List[dict]:
    chunks = list(KNOWLEDGE_BASE)
    social = portfolio_info.get("social", {}) or {}

    chunks.extend([
        {
            "category": "profile",
            "content": (
                f"{portfolio_info.get('name', 'Varun')} is a {portfolio_info.get('title', 'Data Scientist & ML Engineer')} "
                f"focused on {portfolio_info.get('field_of_study', 'Computer Science')}. "
                f"Bio: {portfolio_info.get('bio', '')} Location: {portfolio_info.get('location', '')}."
            ),
        },
        {
            "category": "collaboration",
            "content": (
                "For collaboration, portfolio visitors can contact Varun by email, phone, LinkedIn, or GitHub. "
                f"Email: {portfolio_info.get('email', '')}. Phone: {portfolio_info.get('phone', '')}. "
                f"LinkedIn: {social.get('linkedin', '')}. GitHub: {social.get('github', '')}."
            ),
        },
        {
            "category": "career_goals",
            "content": (
                "Career goals inferred from the portfolio: grow as a data scientist and ML engineer, build practical AI systems, "
                "turn raw data into actionable insights, and work on machine learning, analytics, recommendation, and dashboard projects."
            ),
        },
    ])

    for category, skill_list in (portfolio_info.get("skills", {}) or {}).items():
        chunks.append({
            "category": "skills",
            "content": f"{category.replace('_', ' ').title()} skills: {format_list(skill_list)}.",
        })

    for entry in portfolio_info.get("education", []) or []:
        chunks.append({
            "category": "education",
            "content": (
                f"Education: {entry.get('degree')} at {entry.get('institution')} in {entry.get('location')} "
                f"({entry.get('period')}). Coursework: {format_list(entry.get('coursework'))}."
            ),
        })

    for entry in portfolio_info.get("certifications", []) or []:
        chunks.append({
            "category": "certifications",
            "content": f"Certification: {entry.get('name')} from {entry.get('institution')} ({entry.get('period')}).",
        })

    for project in projects:
        chunks.append({
            "category": "projects",
            "content": (
                f"Project: {project.get('title')}. {project.get('description')} "
                f"Technologies: {format_list(project.get('technologies'))}. "
                f"Category: {project.get('category', 'project')}. Year: {project.get('year') or 'Not listed'}."
            ),
        })

    if resume_info:
        chunks.append({
            "category": "resume",
            "content": (
                f"Resume: {resume_info.get('title', 'Resume')} is available at {resume_info.get('url', '')}. "
                f"File name: {resume_info.get('filename', '')}. Uploaded: {resume_info.get('uploaded_at') or 'default resume'}."
            ),
        })

    return [chunk for chunk in chunks if chunk.get("content")]

def retrieve_relevant_chunks(message: str, chunks: List[dict], limit: int = 6) -> List[dict]:
    if not chunks:
        return []

    corpus = [chunk["content"] for chunk in chunks]
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)
        matrix = vectorizer.transform(corpus + [message]).astype(np.float32)
        dense = normalize(matrix, norm="l2", copy=False).toarray().astype("float32")
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(dense[:-1])
        scores, indices = index.search(dense[-1:].copy(), min(limit, len(chunks)))
        ranked = []
        seen_content = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(chunks[idx])
            normalized_content = re.sub(r"\s+", " ", chunk.get("content", "").strip().lower())
            if normalized_content in seen_content:
                continue
            seen_content.add(normalized_content)
            chunk["score"] = float(score)
            ranked.append(chunk)
        return ranked
    except Exception as exc:
        logger.warning(f"Vector retrieval failed, using keyword retrieval: {exc}")
        terms = set(re.findall(r"[a-z0-9]+", message.lower()))
        ranked = []
        seen_content = set()
        for chunk in chunks:
            content_terms = set(re.findall(r"[a-z0-9]+", chunk["content"].lower()))
            score = len(terms & content_terms)
            if score:
                normalized_content = re.sub(r"\s+", " ", chunk.get("content", "").strip().lower())
                if normalized_content in seen_content:
                    continue
                seen_content.add(normalized_content)
                ranked.append({**chunk, "score": float(score)})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

def clean_chat_response(response: str) -> str:
    lines = [line.rstrip() for line in (response or "").splitlines()]
    cleaned_lines = []
    seen_lines = set()
    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if normalized and normalized in seen_lines:
            continue
        if normalized:
            seen_lines.add(normalized)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def build_contextual_fallback_response(
    message: str,
    portfolio_info: dict,
    projects: List[dict],
    relevant_chunks: List[dict],
    conversation_history: List[dict],
    resume_info: Optional[dict] = None,
) -> str:
    name = portfolio_info.get("name", "Varun")
    intent = detect_chat_intent(message)
    social = portfolio_info.get("social", {}) or {}
    recent_user_questions = [
        item.get("user_message", "")
        for item in conversation_history[-3:]
        if item.get("user_message")
    ]

    selected_chunks = []
    seen_categories = set()
    seen_text = set()
    for chunk in relevant_chunks:
        category = chunk.get("category", "general")
        normalized_text = re.sub(r"\s+", " ", chunk.get("content", "").strip().lower())
        if normalized_text in seen_text:
            continue
        if category in seen_categories and category not in {"projects", "skills"}:
            continue
        if category == "profile" and any(item.get("category") == "about" for item in selected_chunks):
            continue
        seen_text.add(normalized_text)
        seen_categories.add(category)
        selected_chunks.append(chunk)
        if len(selected_chunks) >= 4:
            break

    chunk_lines = [f"- {chunk['content']}" for chunk in selected_chunks]
    if not chunk_lines:
        chunk_lines = [f"- {line}" for line in build_site_context(portfolio_info, projects).split("\n")[:8] if line]

    if intent == "greeting":
        response = (
            f"Hi! I'm {name}'s portfolio assistant. I can help you explore his projects, skills, resume, education, certifications, experience, and collaboration options.\n"
            "- Try asking: What are your best projects?\n"
            "- Or: Tell me about your experience.\n"
            "- Or: How can I collaborate with you?"
        )
    elif intent == "general" and is_low_context_message(message):
        response = (
            f"I'm here with {name}'s portfolio context. Ask me anything specific about his projects, skills, resume, education, certifications, experience, or contact details."
        )
    elif intent == "general" and (not relevant_chunks or (relevant_chunks[0].get("score", 0) < 0.08)):
        response = (
            f"I may not have enough portfolio context to answer that directly. I can still help with {name}'s projects, skills, resume, education, certifications, experience, career goals, or collaboration details."
        )
    elif intent == "career_goals":
        response = (
            f"{name}'s career direction is centered on becoming a stronger data scientist and ML engineer who builds useful, real-world AI systems.\n"
            "- Short term: keep strengthening Python, SQL, machine learning, data analysis, and visualization through practical projects.\n"
            "- Long term: work on intelligent products that turn messy data into clear decisions, especially recommendation systems, analytics dashboards, and applied AI tools.\n"
            "- Current signal: the portfolio shows consistent work across ML, statistics, Streamlit dashboards, and data storytelling."
        )
    elif intent == "collaboration":
        response = (
            f"You can collaborate with {name} on data science, ML, analytics, dashboarding, or AI portfolio projects.\n"
            f"- Best contact: {portfolio_info.get('email', 'email not listed')}\n"
            f"- Phone: {portfolio_info.get('phone', 'not listed')}\n"
            f"- LinkedIn: {social.get('linkedin', 'not listed')}\n"
            f"- GitHub: {social.get('github', 'not listed')}\n"
            "A good first message would include the project goal, dataset or problem area, timeline, and what kind of help you need."
        )
    elif intent == "projects":
        visible_projects = projects[:10]
        if visible_projects:
            response = f"Here are {name}'s portfolio projects:\n" + "\n".join(
                f"- {project.get('title')} ({project.get('year') or 'Year not listed'}): {project.get('description')} Tech: {format_list(project.get('technologies'))}."
                for project in visible_projects
            )
            if len(projects) > len(visible_projects):
                response += f"\nI found {len(projects)} total projects. Ask for more details on any title and I can focus on it."
        else:
            response = f"{name}'s project details are not listed yet."
    elif intent == "skills":
        skills = portfolio_info.get("skills", {}) or {}
        if skills:
            response = f"{name}'s listed technical skills are:\n" + "\n".join(
                f"- {category.replace('_', ' ').title()}: {format_list(skill_list)}"
                for category, skill_list in skills.items()
            )
        else:
            response = f"{name}'s skills are not listed yet."
    elif intent == "education":
        entries = portfolio_info.get("education", []) or []
        response = f"{name}'s education background:\n" + "\n".join(
            f"- {entry.get('degree')} at {entry.get('institution')}, {entry.get('location')} ({entry.get('period')})"
            for entry in entries
        )
    elif intent == "certifications":
        entries = portfolio_info.get("certifications", []) or []
        response = f"{name}'s certifications:\n" + "\n".join(
            f"- {entry.get('name')} from {entry.get('institution')} ({entry.get('period')})"
            for entry in entries
        )
    elif intent == "experience":
        response = (
            f"{name}'s experience is strongest in hands-on data science and machine learning project work.\n"
            + "\n".join(chunk_lines[:5])
            + "\nHe appears comfortable moving from raw data to models, visual analysis, and simple user-facing tools."
        )
    elif intent == "resume" and resume_info:
        response = (
            f"{name}'s resume is available here:\n"
            f"- Preview: {resume_info.get('url')}\n"
            f"- Download: {resume_info.get('url')}\n"
            f"- File: {resume_info.get('filename')}"
        )
    else:
        response = (
            f"Here is the most relevant portfolio context I found for your question:\n"
            + "\n".join(chunk_lines[:5])
            + "\nAsk me for a deeper summary, collaboration fit, or project-specific details and I can narrow it down."
        )

    if recent_user_questions:
        response += f"\n\nContext note: I used this chat's recent questions to avoid repeating the same answer."
    return response

def build_direct_chat_response(message: str, portfolio_info: dict, projects: List[dict]) -> str:
    query = message.lower()
    name = portfolio_info.get("name", "Varun")
    title = portfolio_info.get("title", "Data Scientist & ML Engineer")
    skills = portfolio_info.get("skills", {}) or {}

    if any(word in query for word in ["contact", "email", "phone", "reach", "connect", "linkedin", "github"]):
        social = portfolio_info.get("social", {}) or {}
        return (
            f"Here are {name}'s contact details from the site:\n"
            f"- Email: {portfolio_info.get('email', 'Not listed')}\n"
            f"- Phone: {portfolio_info.get('phone', 'Not listed')}\n"
            f"- LinkedIn: {social.get('linkedin', 'Not listed')}\n"
            f"- GitHub: {social.get('github', 'Not listed')}"
        )

    if any(word in query for word in ["project", "portfolio", "work", "built", "demo"]):
        featured = [project for project in projects if project.get("featured")] or projects[:4]
        if not featured:
            return f"{name}'s project details are not listed yet."
        lines = [
            f"- {project.get('title')}: {project.get('description')} Tech: {format_list(project.get('technologies'))}."
            for project in featured[:5]
        ]
        return f"Here are {name}'s key projects from the site:\n" + "\n".join(lines)

    if any(word in query for word in ["skill", "technology", "tech", "python", "sql", "machine learning", "ml", "tools"]):
        lines = [
            f"- {category.replace('_', ' ').title()}: {format_list(skill_list)}"
            for category, skill_list in skills.items()
        ]
        return f"{name}'s technical skills are:\n" + "\n".join(lines)

    if any(word in query for word in ["education", "study", "college", "university", "degree", "school"]):
        lines = [
            f"- {entry.get('degree')} at {entry.get('institution')}, {entry.get('location')} ({entry.get('period')})"
            for entry in portfolio_info.get("education", [])
        ]
        return f"{name}'s education listed on the site:\n" + "\n".join(lines)

    if any(word in query for word in ["certificate", "certification", "course", "training"]):
        lines = [
            f"- {entry.get('name')} from {entry.get('institution')} ({entry.get('period')})"
            for entry in portfolio_info.get("certifications", [])
        ]
        return f"{name}'s certifications are:\n" + "\n".join(lines)

    if any(word in query for word in ["about", "who", "intro", "introduction", "yourself", "varun"]):
        return (
            f"Here are {name}'s profile details from the site:\n"
            f"- Role: {title}\n"
            f"- Field: {portfolio_info.get('field_of_study', 'Computer Science')}\n"
            f"- Location: {portfolio_info.get('location', 'India')}\n"
            f"- Bio: {portfolio_info.get('bio', '')}"
        )

    return (
        f"I can help with details from {name}'s portfolio site: about, skills, projects, education, certifications, and contact information. "
        "Try asking: 'Show me the projects', 'What are the skills?', or 'How can I contact Varun?'"
    )

def extract_profile_info(portfolio_info: dict) -> dict:
    profile = {}
    for field in PROFILE_FIELDS:
        if field == "social":
            profile[field] = dict(portfolio_info.get(field) or {})
        else:
            profile[field] = portfolio_info.get(field)
    return profile

async def save_portfolio_info(portfolio_info: dict, user_id: str):
    now = datetime.now(timezone.utc).isoformat()
    await db.app_settings.update_one(
        {"_id": "portfolio_info"},
        {
            "$set": {
                "data": portfolio_info,
                "updated_at": now,
                "updated_by": user_id,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

async def push_profile_undo_snapshot(snapshot: dict, user_id: str):
    now = datetime.now(timezone.utc).isoformat()
    undo_doc = await db.app_settings.find_one({"_id": PROFILE_UNDO_DOC_ID})
    history = undo_doc.get("history", []) if undo_doc else []
    history.append({
        "snapshot": snapshot,
        "timestamp": now,
        "updated_by": user_id,
    })
    history = history[-MAX_PROFILE_UNDO_HISTORY:]

    await db.app_settings.update_one(
        {"_id": PROFILE_UNDO_DOC_ID},
        {
            "$set": {
                "history": history,
                "updated_at": now,
                "updated_by": user_id,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

async def pop_profile_undo_snapshot() -> Optional[dict]:
    undo_doc = await db.app_settings.find_one({"_id": PROFILE_UNDO_DOC_ID})
    history = undo_doc.get("history", []) if undo_doc else []
    if not history:
        return None

    entry = history.pop()
    await db.app_settings.update_one(
        {"_id": PROFILE_UNDO_DOC_ID},
        {"$set": {"history": history}},
        upsert=True,
    )
    return entry.get("snapshot")

async def apply_profile_update(update_data: dict, user_id: str, track_undo: bool = True) -> dict:
    current_portfolio = await load_portfolio_info()
    current_profile = extract_profile_info(current_portfolio)

    normalized = dict(update_data)
    if "description" in normalized and "bio" not in normalized:
        normalized["bio"] = normalized.pop("description")
    if "social" in normalized:
        normalized["social"] = {k: v for k, v in (normalized.get("social") or {}).items() if isinstance(v, str)}

    next_portfolio = dict(current_portfolio)
    for key, value in normalized.items():
        if key in PROFILE_FIELDS:
            if key == "social":
                next_portfolio[key] = dict(value or {})
            else:
                next_portfolio[key] = value

    if track_undo:
        await push_profile_undo_snapshot(current_profile, user_id)

    await save_portfolio_info(next_portfolio, user_id)
    return extract_profile_info(next_portfolio)

# ============= PASSWORD & JWT HELPERS =============

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
        "type": "access"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "type": "refresh"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user["_id"] = str(user["_id"])
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_admin_user(request: Request) -> dict:
    user = await get_current_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ============= AUTH ENDPOINTS =============

@api_router.post("/auth/register")
async def register(user_data: UserCreate):
    existing = await db.users.find_one({"email": user_data.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_doc = {
        "email": user_data.email.lower(),
        "password_hash": hash_password(user_data.password),
        "name": user_data.name,
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    access_token = create_access_token(user_id, user_data.email)
    refresh_token = create_refresh_token(user_id)
    
    response = JSONResponse(content={
        "id": user_id,
        "email": user_data.email.lower(),
        "name": user_data.name,
        "role": "user"
    })
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=False, samesite="lax", max_age=3600, path="/")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=False, samesite="lax", max_age=604800, path="/")
    return response

@api_router.post("/auth/login")
async def login(user_data: UserLogin, request: Request):
    user = await db.users.find_one({"email": user_data.email.lower()})
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user_id = str(user["_id"])
    access_token = create_access_token(user_id, user["email"])
    refresh_token = create_refresh_token(user_id)
    
    response = JSONResponse(content={
        "id": user_id,
        "email": user["email"],
        "name": user["name"],
        "role": user["role"]
    })
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=False, samesite="lax", max_age=3600, path="/")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=False, samesite="lax", max_age=604800, path="/")
    return response

@api_router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        "id": user["_id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"]
    }

@api_router.post("/auth/logout")
async def logout():
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return response

@api_router.post("/auth/refresh")
async def refresh_token(request: Request):
    refresh = request.cookies.get("refresh_token")
    if not refresh:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = jwt.decode(refresh, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        access_token = create_access_token(str(user["_id"]), user["email"])
        response = JSONResponse(content={"message": "Token refreshed"})
        response.set_cookie(key="access_token", value=access_token, httponly=True, secure=False, samesite="lax", max_age=3600, path="/")
        return response
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

# ============= PORTFOLIO DATA ENDPOINTS =============

@api_router.get("/home/info")
async def get_home_info():
    return await load_home_info()

@api_router.get("/admin/home/info")
async def get_admin_home_info(user: dict = Depends(get_admin_user)):
    return await load_home_info()

@api_router.put("/admin/home/info")
async def update_admin_home_info(update: HomeInfoUpdate, user: dict = Depends(get_admin_user)):
    current_info = await load_home_info()
    update_data = update.model_dump(exclude_unset=True)
    current_info.update(update_data)

    await db.app_settings.update_one(
        {"_id": "home_info"},
        {
            "$set": {
                "data": current_info,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "updated_by": user["_id"],
            },
            "$setOnInsert": {
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        upsert=True,
    )
    return current_info

@api_router.get("/portfolio/info")
async def get_portfolio_info():
    """Get Varun's portfolio information"""
    return await load_portfolio_info()

@api_router.get("/admin/portfolio/info")
async def get_admin_portfolio_info(user: dict = Depends(get_admin_user)):
    return await load_portfolio_info()

@api_router.put("/admin/portfolio/info")
async def update_admin_portfolio_info(update: PortfolioInfoUpdate, user: dict = Depends(get_admin_user)):
    current_info = await load_portfolio_info()
    update_data = update.model_dump(exclude_unset=True)
    current_info.update(update_data)
    await save_portfolio_info(current_info, user["_id"])
    return current_info

def certificate_filename(original_filename: str, content_type: str) -> str:
    name = os.path.splitext(original_filename or "certificate")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "certificate"
    extension = ALLOWED_CERTIFICATE_TYPES[content_type]
    return f"{slug}-{secrets.token_hex(4)}{extension}"

@api_router.post("/admin/certificates/upload")
async def upload_certificate_file(file: UploadFile = File(...), user: dict = Depends(get_admin_user)):
    if file.content_type not in ALLOWED_CERTIFICATE_TYPES:
        raise HTTPException(status_code=400, detail="Upload a PDF, JPG, PNG, or WEBP certificate")

    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded certificate is empty")
    if len(file_content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Certificate file must be 8MB or smaller")

    os.makedirs(CERTIFICATE_ASSET_DIR, exist_ok=True)
    filename = certificate_filename(file.filename, file.content_type)
    file_path = os.path.join(CERTIFICATE_ASSET_DIR, filename)
    with open(file_path, "wb") as certificate_file:
        certificate_file.write(file_content)

    return {"credential_url": f"{CERTIFICATE_PUBLIC_PATH}/{filename}"}

@api_router.get("/resume/info")
async def get_resume_info():
    return await load_resume_info()

@api_router.get("/admin/resume/info")
async def get_admin_resume_info(user: dict = Depends(get_admin_user)):
    return await load_resume_info()

@api_router.post("/admin/resume/upload")
async def upload_resume_file(file: UploadFile = File(...), user: dict = Depends(get_admin_user)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF resume")

    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty")
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Resume file must be 10MB or smaller")

    os.makedirs(RESUME_ASSET_DIR, exist_ok=True)
    filename = DEFAULT_RESUME_FILENAME
    file_path = os.path.join(RESUME_ASSET_DIR, filename)
    with open(file_path, "wb") as resume_file:
        resume_file.write(file_content)

    resume_info = {
        "title": os.path.splitext(file.filename or "Resume")[0].replace("_", " ").replace("-", " ").strip(),
        "filename": file.filename,
        "url": f"{RESUME_PUBLIC_PATH}/{filename}",
        "content_type": file.content_type,
        "size": len(file_content),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_by": user["_id"],
    }
    await save_resume_info(resume_info)
    return resume_info

@api_router.get("/profile/info")
async def get_profile_info():
    portfolio_info = await load_portfolio_info()
    return extract_profile_info(portfolio_info)

@api_router.get("/admin/profile/info")
async def get_admin_profile_info(user: dict = Depends(get_admin_user)):
    portfolio_info = await load_portfolio_info()
    return extract_profile_info(portfolio_info)

@api_router.put("/admin/profile/info")
async def update_admin_profile_info(update: ProfileInfoUpdate, user: dict = Depends(get_admin_user)):
    update_data = update.model_dump(exclude_unset=True)
    return await apply_profile_update(update_data, user["_id"], track_undo=True)

@api_router.post("/admin/profile/undo")
async def undo_admin_profile_update(user: dict = Depends(get_admin_user)):
    previous_snapshot = await pop_profile_undo_snapshot()
    if not previous_snapshot:
        raise HTTPException(status_code=404, detail="No profile changes to undo")
    restored_profile = await apply_profile_update(previous_snapshot, user["_id"], track_undo=False)
    return restored_profile

@api_router.post("/admin/profile/image")
async def upload_profile_image(file: UploadFile = File(...), user: dict = Depends(get_admin_user)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    image_content = await file.read()
    if not image_content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(image_content) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size must be 2MB or smaller")

    encoded_image = base64.b64encode(image_content).decode("ascii")
    image_url = f"data:{file.content_type};base64,{encoded_image}"
    return await apply_profile_update({"profile_image_url": image_url}, user["_id"], track_undo=True)

@api_router.get("/projects")
async def get_projects():
    """Get all projects"""
    projects = await db.projects.find({}).sort([("sort_order", 1), ("created_at", -1)]).to_list(100)
    return [serialize_project(project) for project in projects]

@api_router.post("/projects/restore")
async def restore_project(project: ProjectRestore, user: dict = Depends(get_admin_user)):
    object_id = parse_project_id(project.id)
    if await db.projects.find_one({"_id": object_id}):
        raise HTTPException(status_code=409, detail="Project already exists")

    project_doc = project.model_dump(exclude={"id"})
    project_doc["_id"] = object_id
    project_doc["restored_at"] = datetime.now(timezone.utc).isoformat()
    project_doc["restored_by"] = user["_id"]
    if not project_doc.get("created_at"):
        project_doc["created_at"] = project_doc["restored_at"]

    try:
        await db.projects.insert_one(project_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Project already exists")

    restored = await db.projects.find_one({"_id": object_id})
    return {"message": "Project restored successfully", "project": serialize_project(restored)}

@api_router.get("/projects/{project_id}")
async def get_project(project_id: str):
    project = await db.projects.find_one({"_id": parse_project_id(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return serialize_project(project)

@api_router.post("/projects")
async def create_project(project: ProjectCreate, user: dict = Depends(get_admin_user)):
    project_doc = project.model_dump()
    project_doc["created_at"] = datetime.now(timezone.utc).isoformat()
    project_doc["created_by"] = user["_id"]
    project_doc["sort_order"] = await db.projects.count_documents({})
    result = await db.projects.insert_one(project_doc)
    
    created = await db.projects.find_one({"_id": result.inserted_id})
    return serialize_project(created)

@api_router.put("/projects/{project_id}")
async def update_project(project_id: str, project: ProjectUpdate, user: dict = Depends(get_admin_user)):
    update_data = {k: v for k, v in project.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    object_id = parse_project_id(project_id)
    result = await db.projects.update_one({"_id": object_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    updated = await db.projects.find_one({"_id": object_id})
    return {"message": "Project updated successfully", "project": serialize_project(updated)}

@api_router.delete("/projects/{project_id}")
async def delete_project(project_id: str, user: dict = Depends(get_admin_user)):
    object_id = parse_project_id(project_id)
    project = await db.projects.find_one({"_id": object_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await db.projects.delete_one({"_id": object_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully", "project": serialize_project(project)}

# ============= CONTACT ENDPOINT =============

@api_router.post("/contact")
async def submit_contact(message: ContactMessage):
    contact_doc = message.model_dump()
    contact_doc["created_at"] = datetime.now(timezone.utc).isoformat()
    contact_doc["status"] = "unread"
    await db.contact_messages.insert_one(contact_doc)
    return {"message": "Message sent successfully! I'll get back to you soon."}

@api_router.get("/contact/messages")
async def get_contact_messages(user: dict = Depends(get_admin_user)):
    messages = await db.contact_messages.find({}).sort("created_at", -1).to_list(100)
    return [serialize_contact_message(message) for message in messages]

@api_router.delete("/contact/messages/{message_id}")
async def delete_contact_message(message_id: str, user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(message_id):
        raise HTTPException(status_code=404, detail="Message not found")

    result = await db.contact_messages.delete_one({"_id": ObjectId(message_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"message": "Contact message deleted successfully"}

# ============= RAG CHATBOT ENDPOINTS =============

# Knowledge base for RAG
KNOWLEDGE_BASE = [
    {"category": "about", "content": "Varun is a Data Scientist and B.Tech Computer Science student at Sardar Beant Singh State University, Gurdaspur. He is passionate about transforming raw data into meaningful insights."},
    {"category": "skills", "content": "Varun is proficient in Python, SQL, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Streamlit. He has expertise in Machine Learning, Statistical Analysis, and Data Visualization."},
    {"category": "projects", "content": "Varun has worked on Netflix Recommendation System using collaborative filtering, COVID-19 Data Analysis with trend forecasting, and Disease & Environment Correlation Analysis with interactive dashboards."},
    {"category": "education", "content": "Varun is pursuing B.Tech in Computer Science from SBSSU (2022-2026), Advanced AI/Data Science training from Intellipaat (2025-2026), and Data Science training from Alpha IT Managed Services (Jan 2026 - July 2026, Remote). He completed high school from Kendriya Vidyalaya."},
    {"category": "contact", "content": "You can reach Varun at varunsharma1234566@gmail.com or +91 6239753187. Connect on LinkedIn: linkedin.com/in/varun-sharma-4525b1343 or GitHub: github.com/varun72004"},
    {"category": "experience", "content": "Varun has hands-on experience as a Data Scientist Intern, working on collecting, cleaning, and interpreting large datasets. He has built recommendation systems and data analysis pipelines."},
    {"category": "certifications", "content": "Varun holds certifications in Microsoft SQL Certification Training from Intellipaat, Python Certification Course from Intellipaat, Data Science from Coder Roots, and Industrial Training in Python from Tech World Institute."},
    {"category": "location", "content": "Varun is based in Pathankot, Punjab, India."},
]

@api_router.post("/chat")
async def chat_with_bot(chat: ChatMessage):
    """Contextual RAG chatbot endpoint with vector retrieval and memory."""
    session_id = chat.session_id or secrets.token_hex(16)
    portfolio_info = await load_portfolio_info()
    resume_info = await load_resume_info()
    projects = await db.projects.find({}).sort("sort_order", 1).to_list(100)
    serialized_projects = [serialize_project(project) for project in projects]
    prev_messages = await db.chatbot_conversations.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(8).to_list(8)
    prev_messages.reverse()
    knowledge_chunks = build_knowledge_chunks(portfolio_info, serialized_projects, resume_info)
    relevant_chunks = retrieve_relevant_chunks(chat.message, knowledge_chunks, limit=7)
    intent = detect_chat_intent(chat.message)

    if intent == "greeting" or (intent == "general" and is_low_context_message(chat.message)):
        response = clean_chat_response(build_contextual_fallback_response(
            chat.message,
            portfolio_info,
            serialized_projects,
            [],
            prev_messages,
            resume_info,
        ))
        await db.chatbot_conversations.insert_one({
            "session_id": session_id,
            "user_message": chat.message,
            "bot_response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "retrieved_categories": [],
            "mode": "direct_intent",
        })
        await db.analytics_logs.insert_one({
            "event_type": "chatbot_interaction",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"intent": intent, "mode": "direct_intent"}
        })
        return {
            "response": response,
            "session_id": session_id,
            "intent": intent,
            "sources": [],
        }

    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage

        conversation_history = ""
        if prev_messages:
            for msg in prev_messages:
                conversation_history += f"User: {msg.get('user_message', '')}\nAssistant: {msg.get('bot_response', '')}\n"

        retrieved_context = "\n".join(
            f"- [{chunk.get('category')}] {chunk.get('content')}"
            for chunk in relevant_chunks
        )

        system_message = f"""You are Varun's AI Portfolio Assistant. You help visitors learn about Varun using only the portfolio site data below.

RETRIEVED SITE CONTEXT:
{retrieved_context}

PREVIOUS CONVERSATION:
{conversation_history}

DETECTED INTENT:
{intent}

INSTRUCTIONS:
- Be friendly, professional, and human-like.
- Answer only from RETRIEVED SITE CONTEXT and PREVIOUS CONVERSATION. Do not invent employers, degrees, dates, or private details.
- If the user asks about career goals, collaboration, or experience, synthesize a useful answer from the listed skills, projects, education, and contact data.
- Use short markdown: one concise paragraph plus bullets when helpful.
- Avoid repeating the previous assistant answer. Rephrase and add a new angle when the user asks a related question.
- Mention exact project, certificate, skill, education, resume, and contact details when relevant.
- If the site data cannot answer the question, say what is available and invite the user to ask about portfolio topics."""

        llm = LlmChat(
            api_key=os.environ.get("EMERGENT_LLM_KEY"),
            session_id=session_id,
            system_message=system_message
        ).with_model("openai", os.environ.get("CHATBOT_MODEL", "gpt-4o-mini"))
        
        user_msg = UserMessage(text=chat.message)
        response = clean_chat_response(await llm.send_message(user_msg))
        if prev_messages and response.strip() == (prev_messages[-1].get("bot_response") or "").strip():
            response = clean_chat_response(build_contextual_fallback_response(
                chat.message,
                portfolio_info,
                serialized_projects,
                relevant_chunks,
                prev_messages,
                resume_info,
            ))
        
        # Store conversation
        await db.chatbot_conversations.insert_one({
            "session_id": session_id,
            "user_message": chat.message,
            "bot_response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "retrieved_categories": [chunk.get("category") for chunk in relevant_chunks],
            "mode": "llm_rag",
        })
        
        # Track analytics
        await db.analytics_logs.insert_one({
            "event_type": "chatbot_interaction",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"intent": intent, "mode": "llm_rag"}
        })
        
        return {
            "response": response,
            "session_id": session_id,
            "intent": intent,
            "sources": [chunk.get("category") for chunk in relevant_chunks],
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        response = clean_chat_response(build_contextual_fallback_response(
            chat.message,
            portfolio_info,
            serialized_projects,
            relevant_chunks,
            prev_messages,
            resume_info,
        ))
        await db.chatbot_conversations.insert_one({
            "session_id": session_id,
            "user_message": chat.message,
            "bot_response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "retrieved_categories": [chunk.get("category") for chunk in relevant_chunks],
            "mode": "local_rag",
        })
        await db.analytics_logs.insert_one({
            "event_type": "chatbot_interaction",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"intent": intent, "mode": "local_rag"}
        })
        return {
            "response": response,
            "session_id": session_id,
            "intent": intent,
            "sources": [chunk.get("category") for chunk in relevant_chunks],
        }

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    history = await db.chatbot_conversations.find(
        {"session_id": session_id}, {"_id": 0}
    ).sort("timestamp", 1).to_list(50)
    return history

# ============= VOICE ASSISTANT ENDPOINTS =============

@api_router.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Speech-to-text using Whisper"""
    try:
        from emergentintegrations.llm.openai import OpenAISpeechToText
        
        audio_content = await file.read()
        
        stt = OpenAISpeechToText(api_key=os.environ.get("EMERGENT_LLM_KEY"))
        
        response = await stt.transcribe(
            file=io.BytesIO(audio_content),
            model="whisper-1",
            response_format="json"
        )
        
        return {"text": response.text}
    except ImportError:
        logger.error("Transcription error: emergentintegrations package not available")
        raise HTTPException(status_code=501, detail="Speech-to-text feature is not available")
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

@api_router.post("/voice/synthesize")
async def synthesize_speech(text: str):
    """Text-to-speech"""
    try:
        from emergentintegrations.llm.openai import OpenAITextToSpeech
        
        tts = OpenAITextToSpeech(api_key=os.environ.get("EMERGENT_LLM_KEY"))
        
        audio_bytes = await tts.generate_speech(
            text=text,
            model="tts-1",
            voice="nova"
        )
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except ImportError:
        logger.error("TTS error: emergentintegrations package not available")
        raise HTTPException(status_code=501, detail="Text-to-speech feature is not available")
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to synthesize speech")

# ============= FACE RECOGNITION ENDPOINTS =============

@api_router.post("/face/register")
async def register_face(data: FaceRegister, user: dict = Depends(get_current_user)):
    """Register face for a user using real deep learning face embeddings"""
    try:
        # Decode base64 image
        try:
            header, encoded = data.face_data.split(",", 1) if "," in data.face_data else ("", data.face_data)
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            logger.error(f"Failed to decode base64 face data: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 face data format")
            
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode the registered face image")
            
        try:
            from deepface import DeepFace
            objs = DeepFace.represent(img_path=img, model_name='VGG-Face', enforce_detection=False)
            if not objs or len(objs) == 0:
                raise HTTPException(status_code=400, detail="No face or embedding could be extracted from registration image")
            embedding = objs[0]["embedding"]
        except Exception as e:
            logger.error(f"DeepFace registration embedding extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract face embedding: {str(e)}")
            
        # Store face data
        await db.face_embeddings.update_one(
            {"user_id": user["_id"], "model_name": "VGG-Face", "filename": "webcam_registration.jpeg"},
            {"$set": {
                "user_id": user["_id"],
                "filename": "webcam_registration.jpeg",
                "embedding": embedding,
                "model_name": "VGG-Face",
                "created_at": datetime.now(timezone.utc).isoformat()
            }},
            upsert=True
        )

        # Physically save the image to "face recognition photos" directory so it persists in git / deployments!
        try:
            photos_dir = r"c:\Users\Varun\OneDrive\Dokumen\AI portfolio\face recognition photos"
            if not os.path.exists(photos_dir):
                backend_parent = os.path.dirname(os.path.abspath(__file__))
                photos_dir = os.path.join(os.path.dirname(backend_parent), "face recognition photos")
            
            if os.path.exists(photos_dir):
                file_path = os.path.join(photos_dir, "webcam_registration.jpeg")
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Successfully saved registered webcam face physically to: {file_path}")
            else:
                logger.warning(f"Could not locate face recognition photos directory to save the file physically.")
        except Exception as disk_err:
            logger.error(f"Failed to save webcam face physically to disk: {str(disk_err)}")
        
        return {"message": "Face registered successfully with VGG-Face embeddings", "face_id": user["_id"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to register face: {str(e)}")

@api_router.get("/face/templates")
async def get_face_templates(user: dict = Depends(get_current_user)):
    """Retrieve all registered face templates for the authenticated user"""
    try:
        cursor = db.face_embeddings.find({"user_id": user["_id"]}, {"embedding": 0})
        templates = await cursor.to_list(100)
        formatted_templates = []
        for t in templates:
            t["id"] = str(t["_id"])
            del t["_id"]
            formatted_templates.append(t)
        return formatted_templates
    except Exception as e:
        logger.error(f"Failed to fetch face templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch face templates: {str(e)}")

@api_router.delete("/face/templates/{filename}")
async def delete_face_template(filename: str, user: dict = Depends(get_current_user)):
    """Delete a specific face template for the authenticated user"""
    try:
        doc = await db.face_embeddings.find_one({"user_id": user["_id"], "filename": filename})
        if not doc:
            raise HTTPException(status_code=404, detail="Face template not found")
        
        await db.face_embeddings.delete_one({"user_id": user["_id"], "filename": filename})
        return {"message": f"Face template '{filename}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete face template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete face template: {str(e)}")


@api_router.post("/face/login")
async def face_login(data: FaceRegister):
    """Login using face recognition with deep learning embeddings comparison"""
    try:
        # Decode base64 image captured by webcam
        try:
            header, encoded = data.face_data.split(",", 1) if "," in data.face_data else ("", data.face_data)
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            logger.error(f"Failed to decode base64 face data: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 face data format")

        # Load DeepFace
        try:
            from deepface import DeepFace
        except ImportError:
            logger.error("DeepFace package is not installed.")
            raise HTTPException(status_code=501, detail="DeepFace recognition library is not available on server")

        # Convert image bytes to a numpy array for OpenCV
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image from bytes using OpenCV")
            raise HTTPException(status_code=400, detail="Could not decode the captured webcam image")

        # Extract VGG-Face embedding for captured image
        try:
            logger.info("Extracting VGG-Face embedding for login capture...")
            objs = DeepFace.represent(img_path=img, model_name='VGG-Face', enforce_detection=False)
            if not objs or len(objs) == 0:
                raise HTTPException(status_code=400, detail="No face or embedding could be extracted from webcam capture")
            captured_embedding = objs[0]["embedding"]
        except Exception as e:
            logger.error(f"DeepFace embedding extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract face embedding: {str(e)}")

        # Retrieve stored face templates from DB
        templates_cursor = db.face_embeddings.find({"model_name": "VGG-Face"})
        templates = await templates_cursor.to_list(100)
        
        if not templates:
            logger.warning("No registered face templates (embeddings) found in database.")
            raise HTTPException(status_code=404, detail="No registered face profiles found on this server. Please seed templates first.")

        # Calculate cosine distances between captured face embedding and all stored templates
        captured_vector = np.array(captured_embedding)
        captured_norm = np.linalg.norm(captured_vector)

        best_distance = 1.0
        matching_template = None

        for template in templates:
            stored_vector = np.array(template["embedding"])
            stored_norm = np.linalg.norm(stored_vector)
            
            if captured_norm == 0 or stored_norm == 0:
                continue
                
            dot_product = np.dot(captured_vector, stored_vector)
            cosine_similarity = dot_product / (captured_norm * stored_norm)
            cosine_distance = 1.0 - cosine_similarity
            
            logger.info(f"Cosine distance to template '{template.get('filename', 'unknown')}': {cosine_distance:.4f}")
            if cosine_distance < best_distance:
                best_distance = cosine_distance
                matching_template = template

        # VGG-Face recommended threshold for cosine distance is 0.40.
        # If best distance is below 0.40, we have a match!
        THRESHOLD = 0.40
        if matching_template and best_distance < THRESHOLD:
            matched_user_id = matching_template["user_id"]
            user = await db.users.find_one({"_id": ObjectId(matched_user_id)})
            if not user:
                logger.error(f"User associated with matched face not found: {matched_user_id}")
                raise HTTPException(status_code=404, detail="Matched user account not found")

            # Generate JWT cookies and log in!
            user_id = str(user["_id"])
            access_token = create_access_token(user_id, user["email"])
            refresh_token = create_refresh_token(user_id)
            
            response = JSONResponse(content={
                "id": user_id,
                "email": user["email"],
                "name": user["name"],
                "role": user["role"],
                "message": f"Face login successful (Matched {matching_template.get('filename')} with distance {best_distance:.4f})"
            })
            response.set_cookie(key="access_token", value=access_token, httponly=True, secure=False, samesite="lax", max_age=3600, path="/")
            response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=False, samesite="lax", max_age=604800, path="/")
            logger.info(f"Successful face login for user: {user['email']} (matched {matching_template.get('filename')} dist={best_distance:.4f})")
            return response
        else:
            logger.warning(f"Face verification failed. Best cosine distance was {best_distance:.4f} (threshold: {THRESHOLD})")
            raise HTTPException(
                status_code=401, 
                detail=f"Face recognition verification failed. Face does not match the authorized profile (distance: {best_distance:.4f})."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected face login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during face verification: {str(e)}")

# ============= ANALYTICS ENDPOINTS =============

@api_router.post("/analytics/track")
async def track_event(event: AnalyticsEvent, request: Request):
    """Track analytics event"""
    event_doc = event.model_dump()
    event_doc["timestamp"] = datetime.now(timezone.utc).isoformat()
    event_doc["ip"] = request.client.host if request.client else "unknown"
    event_doc["user_agent"] = request.headers.get("user-agent", "unknown")
    await db.analytics_logs.insert_one(event_doc)
    return {"message": "Event tracked"}

@api_router.get("/analytics/summary")
async def get_analytics_summary(user: dict = Depends(get_admin_user)):
    """Get analytics summary for admin"""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_start = (now - timedelta(days=7)).isoformat()
    month_start = (now - timedelta(days=30)).isoformat()
    
    # Total visits
    total_visits = await db.analytics_logs.count_documents({"event_type": "page_view"})
    
    # Today's visits
    today_visits = await db.analytics_logs.count_documents({
        "event_type": "page_view",
        "timestamp": {"$gte": today_start}
    })
    
    # Chatbot interactions
    chatbot_interactions = await db.analytics_logs.count_documents({"event_type": "chatbot_interaction"})
    
    # Page views breakdown
    page_views = await db.analytics_logs.aggregate([
        {"$match": {"event_type": "page_view"}},
        {"$group": {"_id": "$page", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]).to_list(10)
    
    # Recent activity
    recent_activity = await db.analytics_logs.find(
        {}, {"_id": 0}
    ).sort("timestamp", -1).limit(20).to_list(20)
    
    return {
        "total_visits": total_visits,
        "today_visits": today_visits,
        "chatbot_interactions": chatbot_interactions,
        "page_views": page_views,
        "recent_activity": recent_activity
    }

# ============= ADMIN ENDPOINTS =============

@api_router.get("/admin/users")
async def get_users(user: dict = Depends(get_admin_user)):
    users = await db.users.find({}, {"password_hash": 0}).to_list(100)
    for u in users:
        u["_id"] = str(u["_id"])
    return users

@api_router.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, user: dict = Depends(get_admin_user)):
    if user_id == user["_id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    result = await db.users.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

@api_router.post("/admin/knowledge-base")
async def add_knowledge(entry: KnowledgeBaseEntry, user: dict = Depends(get_admin_user)):
    """Add entry to knowledge base"""
    doc = entry.model_dump()
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    await db.knowledge_base.insert_one(doc)
    return {"message": "Knowledge entry added successfully"}

@api_router.get("/admin/knowledge-base")
async def get_knowledge_base(user: dict = Depends(get_admin_user)):
    entries = await db.knowledge_base.find({}, {"_id": 0}).to_list(100)
    return entries

# ============= ROOT ENDPOINT =============

@api_router.get("/")
async def root():
    return {"message": "Varun's AI Portfolio API", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Include router
app.include_router(api_router)

# CORS Configuration
frontend_url = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:3000')
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def seed_admin_face_embeddings(admin_id: str):
    logger.info("Starting admin face recognition embeddings seeding...")
    try:
        from deepface import DeepFace
        photos_dir = r"c:\Users\Varun\OneDrive\Dokumen\AI portfolio\face recognition photos"
        if not os.path.exists(photos_dir):
            backend_parent = os.path.dirname(os.path.abspath(__file__))
            photos_dir = os.path.join(os.path.dirname(backend_parent), "face recognition photos")
            
        if not os.path.exists(photos_dir):
            logger.warning(f"Admin photos directory not found at {photos_dir}. Skipping face embedding seeding.")
            return

        supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        files = [f for f in os.listdir(photos_dir) if os.path.splitext(f.lower())[1] in supported_extensions]
        
        if not files:
            logger.warning(f"No face images found in {photos_dir}.")
            return

        seeded_count = 0
        for filename in files:
            file_path = os.path.join(photos_dir, filename)
            # Check if this file has already been seeded for this admin
            existing = await db.face_embeddings.find_one({
                "user_id": admin_id, 
                "filename": filename, 
                "model_name": "VGG-Face"
            })
            if existing:
                logger.info(f"Face embedding for {filename} already exists in DB.")
                continue
            
            logger.info(f"Computing face embedding for reference image: {filename}...")
            try:
                # Represent returns a list of dicts, each with 'embedding' key
                objs = DeepFace.represent(img_path=file_path, model_name='VGG-Face', enforce_detection=False)
                if not objs or len(objs) == 0:
                    logger.warning(f"Could not extract face representation from {filename}.")
                    continue
                
                embedding = objs[0]["embedding"]
                await db.face_embeddings.update_one(
                    {"user_id": admin_id, "filename": filename, "model_name": "VGG-Face"},
                    {"$set": {
                        "user_id": admin_id,
                        "filename": filename,
                        "embedding": embedding,
                        "model_name": "VGG-Face",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }},
                    upsert=True
                )
                seeded_count += 1
                logger.info(f"Successfully seeded embedding for {filename}.")
            except Exception as e:
                logger.error(f"Error computing embedding for {filename}: {str(e)}")
                
        logger.info(f"Finished seeding admin face embeddings. Total newly seeded: {seeded_count}")
    except Exception as e:
        logger.error(f"Failed to seed admin face embeddings: {str(e)}")

# Startup event - Seed admin
@app.on_event("startup")
async def startup_event():
    global db
    try:
        await client.admin.command("ping")
        logger.info("Successfully connected to MongoDB.")
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection failed: {e}")
        db = create_memory_database()

    # Create indexes
    await db.users.create_index("email", unique=True)
    await db.analytics_logs.create_index("timestamp")
    await db.chatbot_conversations.create_index("session_id")
    await db.projects.create_index("source_key", sparse=True)
    await db.projects.create_index("sort_order")
    await db.face_embeddings.create_index([("user_id", 1), ("filename", 1), ("model_name", 1)])
    await seed_default_portfolio_info()
    await seed_default_home_info()
    await seed_default_projects()
    
    # Seed admin user
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@varunportfolio.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "VarunAdmin@2026")
    
    existing = await db.users.find_one({"email": admin_email})
    if not existing:
        admin_doc = {
            "email": admin_email,
            "password_hash": hash_password(admin_password),
            "name": "Varun (Admin)",
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        result = await db.users.insert_one(admin_doc)
        admin_id = str(result.inserted_id)
        logger.info(f"Admin user created: {admin_email}")
    else:
        admin_id = str(existing["_id"])
        # Update password if changed
        if not verify_password(admin_password, existing["password_hash"]):
            await db.users.update_one(
                {"email": admin_email},
                {"$set": {"password_hash": hash_password(admin_password)}}
            )
            logger.info("Admin password updated")

    # Seed face recognition embeddings for admin from reference photos
    await seed_admin_face_embeddings(admin_id)
    
    # Write test credentials
    os.makedirs(MEMORY_DIR, exist_ok=True)
    credentials_path = os.path.join(MEMORY_DIR, "test_credentials.md")
    credentials_content = f"""# Test Credentials

## Admin Account
- Email: {admin_email}
- Password: {admin_password}
- Role: admin

## Auth Endpoints
- POST /api/auth/login
- POST /api/auth/register
- GET /api/auth/me
- POST /api/auth/logout
"""
    try:
        if not os.path.exists(credentials_path):
            with open(credentials_path, "w") as f:
                f.write(credentials_content)
            logger.info(f"Test credentials written to {credentials_path}")
    except PermissionError as exc:
        logger.warning(f"Could not write test credentials to {credentials_path}: {exc}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)