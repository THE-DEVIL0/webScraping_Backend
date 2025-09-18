from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from bson import ObjectId
from pydantic import HttpUrl

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, handler):
        return {"type": "string"}

class ScrapingSession(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str
    platform: str
    url: str
    max_products: int
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_products: int = 0
    metadata: Dict[str, Any] = {}

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

class Product(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str
    product_name: str
    product_url: str
    platform: str
    images: List[HttpUrl] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, HttpUrl: str}  # Added HttpUrl: str
    }

class BackgroundRemovalTask(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    task_id: str
    input_urls: List[HttpUrl] = []
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    output_urls: List[HttpUrl] = []
    add_white_bg: bool = False
    metadata: Dict[str, Any] = {}

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, HttpUrl: str}  # Added HttpUrl: str
    }

class GenerationTask(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    task_id: str
    prompt: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    output_urls: List[HttpUrl] = []
    metadata: Dict[str, Any] = {}

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, HttpUrl: str}  # Added HttpUrl: str
    }

class OptimizationTask(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    task_id: str
    input_urls: List[HttpUrl] = []
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    output_urls: List[HttpUrl] = []
    metadata: Dict[str, Any] = {}

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, HttpUrl: str}  # Added HttpUrl: str
    }