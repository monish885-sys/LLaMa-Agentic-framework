
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from pydantic import BaseModel, field_validator
import pandas as pd
from collections import defaultdict
from hashlib import md5
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import hashlib
import re
import logging
from pathlib import Path 
import asyncio
import ast
from typing import List, Optional, Union
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
VALID_CHECK_TYPES = {
    "non_empty", "length", "contains", "equals", "in_set", "list_length",
    "unique_list", "step_count", "starts_with", "no_formatting_errors",
    "in_list", "unit_consistency", "subparts", "no_subparts", "is_digit",
    "range", "cognitive_match", "difficulty_match", "mcq_format",
    "subject_topic_match", "question_type_consistency", "no_repetition",
    "conceptual_correctness", "curriculum_alignment", "distractor_quality",
    "visual_aid_suggestion", "learning_outcome_match"
}
VALID_CURRICULUM_VALUES = ["CBSE", "ICSE", "IB", "State", "International"]
COGNITIVE_SKILLS = ["Factual", "Understanding", "Application"]
VALID_DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
import os
import json
import hashlib
import uuid
import io
import pandas as pd
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict
from pathlib import Path
import logging
import openai

from openai import OpenAI
from dotenv import load_dotenv
