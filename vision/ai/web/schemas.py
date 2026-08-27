"""Request models shared by FastAPI domain routers."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    train: int = Field(default=70, ge=1, le=100)
    val: int = Field(default=15, ge=0, le=100)
    test: int = Field(default=15, ge=0, le=100)


class RoboflowRequest(BaseModel):
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    version: Optional[str] = None
    classes: list[str] = Field(default_factory=list)
    name: str = "dataset"
    train: int = Field(default=70, ge=1, le=100)
    val: int = Field(default=15, ge=0, le=100)
    test: int = Field(default=15, ge=0, le=100)
    force_split: bool = False
    job_id: str = ""


class DatasetDownloadRequest(BaseModel):
    dataset_yaml: str


class AnnotationQaRequest(BaseModel):
    dataset_yaml: str
    task: str = "auto"
    model: str = os.getenv("SAM_QA_MODEL", "sam2.1_s.pt")
    scope: str = "all"
    preset: str = "balanced"
    box_tolerance_percent: float = Field(default=5.0, ge=0.0, le=50.0)
    sam_max_difference_percent: float = Field(default=25.0, ge=0.0, le=100.0)
    auto_correction_mode: str = "shadow"
    sam_prompt_expansion_percent: float = Field(default=8.0, ge=0.0, le=50.0)
    sam_prompt_jitter_percent: float = Field(default=2.0, ge=0.0, le=20.0)
    sam_stability_bbox_iou_min: float = Field(default=0.90, ge=0.0, le=1.0)
    sam_stability_edge_percent_max: float = Field(default=3.0, ge=0.0, le=50.0)
    sam_auto_quality_min: float = Field(default=0.85, ge=0.0, le=1.0)
    sam_auto_yolo_iou_min: float = Field(default=0.70, ge=0.0, le=1.0)
    sam_auto_center_shift_max: float = Field(default=0.10, ge=0.0, le=1.0)
    sam_auto_neighbor_iou_max: float = Field(default=0.15, ge=0.0, le=1.0)
    auto_audit_percent: float = Field(default=5.0, ge=0.0, le=100.0)
    max_images: Optional[int] = Field(default=None, ge=1)
    max_side: int = Field(default=1280, ge=320, le=4096)


class AnnotationQaMarkRequest(BaseModel):
    issue_id: str
    status: str


class AnnotationQaFixRequest(BaseModel):
    issue_id: str
    fix: str = "sam_box"
    class_id: Optional[int] = Field(default=None, ge=0)


class AnnotationQaRoboflowRequest(BaseModel):
    api_key: Optional[str] = None
    preview_id: str = ""
    confirmed: bool = False


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


class TestArtifactRequest(BaseModel):
    artifact: str
