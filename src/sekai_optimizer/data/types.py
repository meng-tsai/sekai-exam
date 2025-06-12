from pydantic import BaseModel, Field
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


class UsersResponse(BaseModel):
    """Response schema for user profile generation via LangSmith prompts."""

    users: List[User] = Field(description="List of generated user profiles")


class StoriesResponse(BaseModel):
    """Response schema for story concept generation via LangSmith prompts."""

    stories: List[Story] = Field(description="List of generated story concepts")
