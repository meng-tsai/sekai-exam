from pydantic import BaseModel
from typing import List


class Story(BaseModel):
    """Data model for a single story."""

    id: int
    title: str
    intro: str
    tags: List[str]


class User(BaseModel):
    """Data model for a single user profile."""

    user_id: int
    name: str
    profile: str


class Dataset(BaseModel):
    """Data model for the entire synthetic dataset."""

    stories: List[Story]
    users: List[User]
