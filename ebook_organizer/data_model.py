from pydantic import BaseModel, Field
from typing import List, Optional

class SeriesInfo(BaseModel):
    """Data model for book series information."""
    series_name: Optional[str] = None
    series_number: Optional[int] = None

class BookMetadata(BaseModel):
    """The main data model for all enriched book metadata."""
    title: str = Field(..., description="The correct, clean title of the book.")
    authors: List[str] = Field(..., description="A list of the book's authors.")
    publication_year: Optional[int] = Field(None, description="The original publication year.")
    genre: str = Field("Uncategorized", description="The primary genre of the book.")
    goodreads_rating: Optional[float] = Field(None, description="The average Goodreads rating.")
    review_summary: Optional[str] = Field(None, description="A brief, one or two-sentence summary of the general review sentiment.")
    series_info: Optional[SeriesInfo] = Field(None, description="Information about the series, if applicable.")