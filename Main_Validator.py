import os
import json
import re
import hashlib
import uuid
import asyncio
import io
import pandas as pd
import ast
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict
from pathlib import Path
import logging 
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables")

llm = OpenAI(api_key=OPENAI_KEY)
app = FastAPI()

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
CHECKLIST_DIR = Path("checklists")

def get_fallback_checklist() -> List[Dict]:
    """Return a default checklist with required checks."""
    start_time = time.time()
    fallback = [
        {
            "rule_name": "Question Type Consistency",
            "condition": "all",
            "description": "Ensures question type matches format.",
            "validation_criteria": {
                "field": "question",
                "checks": [{"type": "question_type_consistency"}]
            }
        },
        {
            "rule_name": "No Repetition",
            "condition": "all",
            "description": "Ensures no duplicate questions.",
            "validation_criteria": {
                "field": "question",
                "checks": [{"type": "no_repetition"}]
            }
        },
        {
            "rule_name": "Conceptual Correctness",
            "condition": "all",
            "description": "Ensures conceptual accuracy.",
            "validation_criteria": {
                "field": "explanation",
                "checks": [{"type": "conceptual_correctness"}]
            }
        },
        {
            "rule_name": "Curriculum Alignment",
            "condition": "all",
            "description": "Ensures question aligns with curriculum.",
            "validation_criteria": {
                "field": "question",
                "checks": [{"type": "curriculum_alignment"}]
            }
        },
        {
            "rule_name": "Distractor Quality",
            "condition": "question_type == 'mcq'",
            "description": "Ensures MCQ distractors are explained.",
            "validation_criteria": {
                "field": "explanation",
                "checks": [{"type": "distractor_quality"}]
            }
        },
        {
            "rule_name": "Visual Aid Suggestion",
            "condition": "all",
            "description": "Suggests visual aids where applicable.",
            "validation_criteria": {
                "field": "question",
                "checks": [{"type": "visual_aid_suggestion"}]
            }
        },
        {
            "rule_name": "Learning Outcome Match",
            "condition": "all",
            "description": "Ensures learning outcome alignment.",
            "validation_criteria": {
                "field": "question",
                "checks": [{"type": "learning_outcome_match"}]
            }
        },
        {
            "rule_name": "Difficulty Match",
            "condition": "all",
            "description": "Ensures difficulty matches syllabus.",
            "validation_criteria": {
                "field": "difficulty_level",
                "checks": [{"type": "difficulty_match"}]
            }
        }
    ]
    end_time = time.time()
    logger.info(f"get_fallback_checklist completed in {end_time - start_time:.2f} seconds")
    return fallback

def normalize_text(text: any) -> str:
    """Normalize text for repetition checking."""
    start_time = time.time()
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\\+\(+(.+?)\\+\)', r'\1', text).replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text.strip()).lower()
    result = hashlib.md5(text.encode('utf-8')).hexdigest()
    end_time = time.time()
    logger.info(f"normalize_text completed in {end_time - start_time:.2f} seconds")
    return result

def run_check(check: dict, field_value: any, context: dict) -> tuple[bool, str]:
    """Run a single validation check."""
    start_time = time.time()
    check_type = check.get("type")
    if check_type not in VALID_CHECK_TYPES:
        logger.warning(f"Unknown check type: {check_type}")
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return False, f"Unknown check type: {check_type}"

    normalized_value = field_value
    if isinstance(field_value, str):
        normalized_value = re.sub(r'\\+\(+(.+?)\\+\)', r'\1', field_value).replace('\xa0', ' ').strip()
    elif isinstance(field_value, list):
        normalized_value = [re.sub(r'\\+\(+(.+?)\\+\)', r'\1', str(x)).replace('\xa0', ' ').strip() if isinstance(x, str) else x for x in field_value]
        if len(normalized_value) == 1:
            normalized_value = normalized_value[0]

    if check_type == "no_repetition":
        all_questions = context.get("all_questions", [])
        current_uuid = context.get("uuid", "")
        current_hash = normalize_text(str(normalized_value))
        for other_question in all_questions:
            if other_question["uuid"] == current_uuid:
                continue
            other_hash = normalize_text(other_question.get("question", ""))
            if current_hash == other_hash:
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return False, f"Duplicate question detected (UUID {other_question['uuid']})"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "question_type_consistency":
        question_type = str(context.get("question_type", "")).lower()
        options = context.get("options", [])
        answers = str(context.get("correct_option", ""))
        if question_type in ["vsa", "sa", "la"]:
            if options or answers:
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return False, f"Question type '{question_type}' should not have options or answers"
        elif question_type == "mcq":
            if not isinstance(options, list) or not options:
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return False, "MCQ requires options"
            if answers:
                normalized_options = [str(opt).lower().strip() for opt in options]
                if answers.lower().strip() not in normalized_options:
                    end_time = time.time()
                    logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                    return False, f"Answer '{answers}' not in options"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "conceptual_correctness":
        subject = str(context.get("subject", "")).lower()
        topic = str(context.get("topic", "")).lower()
        if subject == "mathematics" and topic == "circles":
            explanation = str(context.get("explanation", "")).lower()
            if "radius" not in explanation:
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return False, "Explanation does not define radius"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "curriculum_alignment":
        curriculum = str(context.get("curriculum", "")).lower()
        grade = str(context.get("grade", "")).lower()
        topic = str(context.get("topic", "")).lower()
        if curriculum == "cbse" and grade == "9" and topic == "circles":
            question = str(context.get("question", "")).lower()
            if "circle" not in question:
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return False, "Question does not align with CBSE Grade 9 Circles syllabus"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "distractor_quality":
        if str(context.get("question_type", "")).lower() != "mcq":
            end_time = time.time()
            logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
            return True, ""
        options = context.get("options", [])
        correct_answer = str(context.get("correct_option", "")).lower()
        explanation = str(context.get("explanation", "")).lower()
        if len(options) != 4:
            end_time = time.time()
            logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
            return False, "MCQ must have 4 options"
        incorrect_options = [opt.lower() for opt in options if opt.lower() != correct_answer]
        if not all(f"{opt} incorrect" in explanation for opt in incorrect_options):
            end_time = time.time()
            logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
            return False, "Explanation does not address incorrect options"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "visual_aid_suggestion":
        topic = str(context.get("topic", "")).lower()
        if topic == "circles":
            if "diagram" not in str(context.get("question", "")).lower() and "diagram" not in str(context.get("explanation", "")).lower():
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return True, "Inclusion of a diagram could enhance understanding"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "learning_outcome_match":
        topic = str(context.get("topic", "")).lower()
        question = str(context.get("question", "")).lower()
        if topic == "circles":
            if "understand" not in question and "explain" not in question:
                end_time = time.time()
                logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
                return True, "Question could better target understanding of circles"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    if check_type == "difficulty_match":
        difficulty = str(context.get("difficulty_level", "")).lower()
        topic = str(context.get("topic", "")).lower()
        question = str(context.get("question", "")).lower()
        if difficulty not in VALID_DIFFICULTY_LEVELS:
            end_time = time.time()
            logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
            return False, f"Invalid difficulty level '{difficulty}'"
        if topic == "circles" and difficulty == "easy" and ("complex" in question or "advanced" in question):
            end_time = time.time()
            logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
            return True, "Question marked as 'Easy' but contains complex concepts"
        if topic == "circles" and difficulty == "hard" and ("complex" not in question and "advanced" not in question):
            end_time = time.time()
            logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
            return False, "Hard difficulty requires complex or advanced concepts"
        end_time = time.time()
        logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
        return True, ""

    end_time = time.time()
    logger.info(f"run_check completed in {end_time - start_time:.2f} seconds")
    return True, ""

async def generate_dynamic_feedback(input_data: dict, errors: list, warnings: list) -> str:
    """Generate dynamic feedback using LLM based on errors and warnings."""
    input_json = json.dumps(input_data, indent=2)
    errors_json = json.dumps(errors, indent=2)
    warnings_json = json.dumps(warnings, indent=2)

    prompt = f"""
You are an advanced educational content evaluator specializing in curriculum-aligned question analysis. Based on the following data, generate an original, insightful, and concise feedback paragraph (under 100 words). Your goal is to help a content creator improve the question's educational value.
- Summarize all key issues, emphasizing conceptual, curriculum, and cognitive alignment.
- Highlight critical errors that block acceptance.
- Suggest improvements for warnings (e.g., add diagrams, clarify distractors) to enhance learning.
- Tailor to {input_data.get('subject', 'unknown')}, {input_data.get('topic', 'unknown')}, Grade {input_data.get('grade', 'unknown')}.
- Include semantic insights (e.g., does it foster {input_data.get('cognitive_level','unknown')} cognitive skills?).
- Paraphrase issues, avoid repeating error/warning text.
- Keep under 100 words.
Example feedback: "The question is well-framed, clear, and uses appropriate mathematical terminology suitable for Grade 9 CBSE students. It aligns effectively with the topic of circles and covers the learning outcomes related to radius comprehension. The options are relevant and plausible distractors, and the single correct answer is clearly identified. The solution accurately defines the radius, and the explanation rightly justifies the correct answer. However, the explanation can be enhanced by briefly clarifying why the other options are incorrect, which would improve conceptual clarity. Additionally, including references to curriculum content and a diagram would strengthen alignment and educational value. Overall, the question is easy and appropriate for its cognitive skill target, with minor areas for improvement noted."
Input Question:
{input_json}
Errors:
{errors_json}
Warnings:
{warnings_json}
Return only the feedback text.
"""
    try:
        response = llm.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        feedback = response.choices[0].message.content.strip()
        logger.info(f"Generated feedback for question UUID {input_data.get('uuid', 'unknown')}: {feedback}")
        return feedback
    except Exception as e:
        logger.error(f"Error generating feedback for question UUID {input_data.get('uuid', 'unknown')}): {str(e)}")
        return "Validation completed. Revise for conceptual accuracy, curriculum alignment, and pedagogical clarity. Address all listed errors and warnings for details."

def extract_checklist_json(subject: str) -> tuple[List[Dict], bool]:
    """Extract checklist JSON, supplement missing rules from fallback."""
    start_time = time.time()
    used_json = False
    fallback_checklist = get_fallback_checklist()
    required_checks = {
        "question_type_consistency", "no_repetition", "conceptual_correctness",
        "curriculum_alignment", "distractor_quality", "visual_aid_suggestion",
        "learning_outcome_match", "difficulty_match"
    }
    validated_rules = []

    if not subject or not subject.isalnum():
        logger.warning(f"Invalid subject '{subject}'; using fallback checklist")
        end_time = time.time()
        logger.info(f"extract_checklist_json completed in {end_time - start_time:.2f} seconds")
        return fallback_checklist, used_json

    json_path = CHECKLIST_DIR / f"{subject.lower()}.json"
    if not json_path.exists():
        logger.warning(f"Checklist JSON for subject '{subject}' not found; using fallback")
        end_time = time.time()
        logger.info(f"extract_checklist_json completed in {end_time - start_time:.2f} seconds")
        return fallback_checklist, used_json

    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            parsed_json = json.load(file)
        if not isinstance(parsed_json, list):
            logger.warning(f"Checklist JSON is not a list: {parsed_json}; using fallback")
            end_time = time.time()
            logger.info(f"extract_checklist_json completed in {end_time - start_time:.2f} seconds")
            return fallback_checklist, used_json

        seen_checks = set()
        for rule in parsed_json:
            if not isinstance(rule, dict):
                logger.warning(f"Skipping invalid rule: {rule}")
                continue
            rule_name = rule.get("rule_name", "Unnamed Rule")
            validation_criteria = rule.get("validation_criteria", {})
            if not isinstance(validation_criteria, dict):
                logger.warning(f"Invalid validation_criteria in rule '{rule_name}'")
                continue
            field = validation_criteria.get("field")
            if not isinstance(field, str) or not field:
                logger.warning(f"Missing field in rule '{rule_name}'")
                continue
            checks = validation_criteria.get("checks", [])
            if not isinstance(checks, list):
                logger.warning(f"Invalid checks in rule '{rule_name}'")
                continue
            valid_checks = [check for check in checks if isinstance(check, dict) and check.get("type") in VALID_CHECK_TYPES]
            if not valid_checks:
                logger.warning(f"No valid checks in rule '{rule_name}'")
                continue
            rule["validation_criteria"]["checks"] = valid_checks
            validated_rules.append(rule)
            seen_checks.update(check.get("type") for check in valid_checks)

        missing_checks = required_checks - seen_checks
        if missing_checks:
            logger.warning(f"Missing checks: {missing_checks}; supplementing from fallback")
            for check_type in missing_checks:
                fallback_rule = next((r for r in fallback_checklist if r["validation_criteria"]["checks"][0]["type"] == check_type), None)
                if fallback_rule:
                    validated_rules.append(fallback_rule)
                    logger.info(f"Added fallback rule for '{check_type}'")

        used_json = bool(validated_rules)
        if not validated_rules:
            logger.warning("No valid rules extracted; using fallback")
            validated_rules = fallback_checklist
        end_time = time.time()
        logger.info(f"extract_checklist_json completed in {end_time - start_time:.2f} seconds")
        return validated_rules, used_json
    except Exception as e:
        logger.error(f"Error parsing checklist JSON for '{subject}': {e}")
        end_time = time.time()
        logger.info(f"extract_checklist_json completed in {end_time - start_time:.2f} seconds")
        return fallback_checklist, used_json

async def apply_criteria(criteria: dict, data: dict) -> tuple[bool, str]:
    """Apply validation criteria to data."""
    start_time = time.time()
    field = criteria.get("field", "")
    checks = criteria.get("checks", [])
    if not field or not checks:
        end_time = time.time()
        logger.info(f"apply_criteria completed in {end_time - start_time:.2f} seconds")
        return False, "Invalid criteria"

    if field not in data or data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
        end_time = time.time()
        logger.info(f"apply_criteria completed in {end_time - start_time:.2f} seconds")
        return True, f"Field '{field}' is missing; skipping"

    for check in checks:
        is_valid, reason = run_check(check, data[field], data)
        if not is_valid:
            end_time = time.time()
            logger.info(f"apply_criteria completed in {end_time - start_time:.2f} seconds")
            return False, reason
    end_time = time.time()
    logger.info(f"apply_criteria completed in {end_time - start_time:.2f} seconds")
    return True, ""

async def rule_based_validate(input_data: dict, all_questions: List[Dict], checklist: List[Dict]) -> dict:
    """Perform strict rule-based validation (Agent 1). Enforces checklist rules rigorously."""
    start_time = time.time()
    max_score = 100
    score = max_score
    errors = []
    warnings = []
    data = {k: v for k, v in input_data.items()}
    data["all_questions"] = all_questions

    critical_check_types = {
        "conceptual_correctness",
        "curriculum_alignment",
        "no_repetition",
        "difficulty_match",
        "question_type_consistency"
    }

    for rule in checklist:
        criteria = rule.get("validation_criteria", {})
        is_valid, reason = await apply_criteria(criteria, data)
        check_type = criteria.get("checks", [{}])[0].get("type", "").lower()

        if not is_valid:
            if check_type in critical_check_types:
                errors.append(reason)
                score -= 25
            else:
                warnings.append(reason)
                score -= 10

    if len(warnings) >= 3:
        errors.append("Too many warnings; question requires revision.")
        score -= 10
        warnings.clear()

    score = max(1, score)

    end_time = time.time()
    logger.info(f"rule_based_validate completed in {end_time - start_time:.2f} seconds")

    return {
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "uuid": data.get("uuid", ""),
        "is_duplicate": any("duplicate" in e.lower() for e in errors)
    }

async def llm_based_validate(input_data: dict, rule_based_result: dict, checklist: List[Dict]) -> dict:
    """Perform LLM-based validation (Agent 2), reviewing Agent 1's output and cross-checking unapplied rules."""
    start_time = time.time()
    score = 100
    errors = []
    warnings = []
    data = {k: v for k, v in input_data.items()}
    rules_text = "\n".join(f"Rule: {r['rule_name']} - {r['description']}" for r in checklist)

    applied_checks = set()
    for error in rule_based_result.get("errors", []):
        for check_type in VALID_CHECK_TYPES:
            if check_type in error.lower():
                applied_checks.add(check_type)
    for warning in rule_based_result.get("warnings", []):
        for check_type in VALID_CHECK_TYPES:
            if check_type in warning.lower():
                applied_checks.add(check_type)

    all_check_types = set(check["type"] for rule in checklist for check in rule.get("validation_criteria", {}).get("checks", []))
    unapplied_checks = all_check_types - applied_checks

    if unapplied_checks:
        missed_checks_prompt = f"""
The following checklist rules were not applied by Agent 1:
{', '.join(unapplied_checks)}
For each unapplied rule, evaluate the question to determine if it passes the check. Provide specific errors or warnings if the check fails.
"""

    prompt = f"""
You are an expert educational content validator. Your role is to critically evaluate a multiple-choice question and the results of a rule-based validator (Agent 1). Use the provided checklist to assess accuracy, alignment, and quality, and cross-check for any missed rules.

Your task:
- Grammatically and semantically validate the question(Highest priority).
- Re-validate the question using your own judgment and the checklist rules below.
- Verify that Agent 1's reported errors and warnings are justified.
- Identify any missed issues (content errors, misalignment, ambiguity) from the checklist and perfom that check.
- Check if the assigned diffuculty level is appropriate for the question's conceptual depth follwing the syllabus.
- Cross-check unapplied rules: {missed_checks_prompt}
- Ensure the 'difficulty_level' matches the conceptual depth ('Easy', 'Medium', 'Hard').
- Identify duplication or repetition patterns.
- Penalize conceptual errors and difficulty mismatches more heavily.
- Do not repeat Agent 1 feedback verbatim—rephrase and refine it.
- Keep the score between 1–100 based on severity of issues.
- Keep output focused and JSON-valid.

Question to Validate:
{json.dumps(data, indent=2)}
Checklist:
{rules_text}
Rule-Based Validation (Agent 1) Result:
{json.dumps(rule_based_result, indent=2)}
Recheck Agent 1's errors and warnings for accuracy. Apply any unapplied rules listed above. Adjust if necessary, ensuring difficulty level is valid ('Easy', 'Medium', 'Hard') and aligns with the syllabus.
Return JSON:
{{
  "score": int,
  "errors": [],
  "warnings": [],
  "uuid": str,
  "is_duplicate": bool
}}
"""
    try:
        response = llm.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a precise validator for educational content, reviewing rule-based outputs and applying missed checks."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())

        # difficulty = str(data.get("difficulty_level", ""))
        # topic = str(data.get("topic", ""))
        # question = str(data.get("question", ""))
        # if topic == "circles" and difficulty == "easy" and ("complex" in question or "advanced" in question):
        #     if not any("complex" in w.lower() for w in result["warnings"]):
        #         result["warnings"].append("Question marked as 'Easy' but contains complex concepts")
        #         result["score"] = max(1, result.get("score", 100) - 5)
        # if topic == "circles" and difficulty == "hard" and ("complex" not in question and "advanced" not in question):
        #     if not any("hard" in e.lower() for e in result["errors"]):
        #         result["errors"].append("Hard difficulty requires complex or advanced concepts")
        #         result["score"] = max(1, result.get("score", 100) - 20)

        if unapplied_checks:
            logger.info(f"Agent 2 reapplied missed checks: {unapplied_checks}")

        end_time = time.time()
        logger.info(f"llm_based_validate completed in {end_time - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        errors.append(f"LLM validation failed: {str(e)}")
        score -= 20
        end_time = time.time()
        logger.info(f"llm_based_validate completed in {end_time - start_time:.2f} seconds")
        return {
            "score": max(1, score),
            "errors": errors,
            "warnings": warnings,
            "uuid": data.get("uuid", ""),
            "is_duplicate": False
        }

async def meta_validate(input_data: dict, rule_based_result: dict, llm_based_result: dict) -> dict:
    """Synthesize validation results using an LLM-based Meta Agent (Agent 3)."""
    start_time = time.time()

    try:
        meta_prompt = f"""
You are a senior educational validation agent. Your task is to synthesize and evaluate two prior validation results of an educational question.

Input Question:
{json.dumps(input_data, indent=2)}

Rule-Based Validation (Agent 1):
{json.dumps(rule_based_result, indent=2)}

LLM-Based Validation (Agent 2):
{json.dumps(llm_based_result, indent=2)}

Your responsibilities:
- Combine both results to determine the final error and warning list.
- Decide on the correct score (1–100) and a validation_score (1–5).
- Set `final_decision` to one of: "ACCEPT", "REVIEW", "REJECT".
- Set `is_valid` to `false` only if the question is a duplicate or has blocking issues.
- Provide specific, practical improvement suggestions in `improvement_recommendations`.

Return a JSON in the following format:
{{
  "is_valid": bool,
  "validation_score": int,
  "errors": [string],
  "warnings": [string],
  "final_decision": string,
  "improvement_recommendations": {{
    "specific_fixes": [string]
  }}
}}
Only return the JSON.
"""
        response = llm.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a highly skilled educational validator synthesizing multi-agent validation outputs."},
                {"role": "user", "content": meta_prompt}
            ],
            response_format={"type": "json_object"}
        )
        meta_output = json.loads(response.choices[0].message.content.strip())

        feedback = await generate_dynamic_feedback(input_data, meta_output.get("errors", []), meta_output.get("warnings", []))

        result = {
            "is_valid": meta_output["is_valid"],
            "data": {
                "is_valid": meta_output["is_valid"],
                "validation_score": meta_output["validation_score"],
                "errors": meta_output.get("errors", []),
                "warnings": meta_output.get("warnings", []),
                "feedback": feedback,
                "improvement_recommendations": meta_output.get("improvement_recommendations", {}),
                "final_decision": meta_output.get("final_decision", "REVIEW")
            }
        }

    except Exception as e:
        logger.error(f"Meta LLM validation failed: {e}")
        errors = list(set(rule_based_result.get("errors", []) + llm_based_result.get("errors", [])))
        warnings = list(set(rule_based_result.get("warnings", []) + llm_based_result.get("warnings", [])))
        is_duplicate = rule_based_result.get("is_duplicate", False) or llm_based_result.get("is_duplicate", False)
        final_decision = "REJECT" if errors else ("REVIEW" if warnings else "ACCEPT")
        validation_score = 1 if errors else (4 if warnings else 5)
        feedback = await generate_dynamic_feedback(input_data, errors, warnings)

        result = {
            "is_valid": not is_duplicate,
            "data": {
                "is_valid": not is_duplicate,
                "validation_score": validation_score,
                "errors": errors,
                "warnings": warnings,
                "feedback": feedback,
                "improvement_recommendations": {
                    "specific_fixes": [f"Address: {e}" for e in errors] + [f"Consider: {w}" for w in warnings]
                },
                "final_decision": final_decision
            }
        }

    end_time = time.time()
    logger.info(f"meta_validate completed in {end_time - start_time:.2f} seconds")
    return result

class QuestionInput(BaseModel):
    question: str
    options: List[str] = []
    correct_option: Optional[str] = None
    solution: Optional[str] = ""
    explanation: Optional[str] = None
    subject: str
    topic: str
    grade: str
    difficulty_level: str
    cognitive_skill: str
    question_type: str
    curriculum: str
    version_uuid: Optional[str] = ""
    uuid: Optional[str] = None
    createdAt: Optional[str] = ""
    rejectReason: Optional[str] = ""

    @field_validator('options', mode='before')
    def parse_options(cls, v):
        if isinstance(v, str):
            try:
                return ast.literal_eval(v) if v.strip().startswith('[') else [opt.strip() for opt in v.split(',') if opt.strip()]
            except Exception:
                return [opt.strip() for opt in v.split(',') if opt.strip()]
        return [str(opt).strip() for opt in v] if v else []

    @field_validator('correct_option', mode='before')
    def parse_answers(cls, v):
        return str(v).strip() if v else None

async def parse_excel(file: UploadFile) -> List[Dict]:
    """Parse Excel file into questions."""
    start_time = time.time()
    try:
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))
        questions = []
        for _, row in df.iterrows():
            options = [str(opt).strip() for opt in ast.literal_eval(row.get("options", "[]"))] if row.get("options") else []
            question = {
                "question": str(row.get("question", "")).replace('\xa0', ' '),
                "options": options,
                "correct_option": str(row.get("correct_option", "")).replace('\xa0', ' '),
                "solution": str(row.get("solution", "")).replace('\xa0', ' '),
                "explanation": str(row.get("explanation", "")).replace('\xa0', ' '),
                "subject": str(row.get("subject")),
                "topic": str(row.get("topic")),
                "grade": str(row.get("grade")),
                "difficulty_level": str(row.get("difficulty_level")),
                "cognitive_skill": str(row.get("cognitive_skill")),
                "question_type": str(row.get("question_type")).lower(),
                "curriculum": str(row.get("curriculum")),
                "version_uuid": str(row.get("version_uuid")),
                "uuid": str(row.get("uuid", str(uuid.uuid4()))),
                "createdAt": str(row.get("createdAt", "")),
                "rejectReason": str(row.get("rejectReason", ""))
            }
            questions.append(question)
        end_time = time.time()
        logger.info(f"parse_excel completed in {end_time - start_time:.2f} seconds")
        return questions
    except Exception as e:
        end_time = time.time()
        logger.error(f"parse_excel failed: {e}")
        logger.info(f"parse_excel failed in {end_time - start_time:.2f} seconds")
        raise HTTPException(status_code=400, detail=f"Invalid Excel file: {e}")

async def parse_json(json_data: Union[dict, list]) -> List[Dict]:
    """Parse JSON data into questions."""
    start_time = time.time()
    try:
        questions = []
        input_questions = [json_data] if isinstance(json_data, dict) else json_data
        for item in input_questions:
            question_input = QuestionInput(**item)
            question = {
                "question": question_input.question,
                "options": question_input.options,
                "correct_option": question_input.correct_option,
                "solution": question_input.solution,
                "explanation": question_input.explanation,
                "subject": question_input.subject,
                "topic": question_input.topic,
                "grade": question_input.grade,
                "difficulty_level": question_input.difficulty_level,
                "cognitive_skill": question_input.cognitive_skill,
                "question_type": question_input.question_type.lower(),
                "curriculum": question_input.curriculum,
                "version_uuid": question_input.version_uuid,
                "uuid": question_input.uuid or str(uuid.uuid4()),
                "createdAt": question_input.createdAt,
                "rejectReason": question_input.rejectReason
            }
            questions.append(question)
        end_time = time.time()
        logger.info(f"parse_json completed in {end_time - start_time:.2f} seconds")
        return questions
    except Exception as e:
        end_time = time.time()
        logger.error(f"parse_json failed: {e}")
        logger.info(f"parse_json failed in {end_time - start_time:.2f} seconds")
        raise HTTPException(status_code=400, detail=f"Failed to parse JSON: {e}")

async def validate_questions(questions: List[Dict]) -> List[Dict]:
    """Validate questions with two iterations."""
    start_time = time.time()
    if not questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    subject = str(questions[0].get("subject", "Mathematics")).strip().lower()
    checklist, used_json = extract_checklist_json(subject)
    checklist_source = "json_file" if used_json else "fallback"
    logger.info(f"Using {checklist_source} checklist for subject: {subject}")

    results = []
    for iteration in range(2):
        logger.info(f"Starting validation iteration {iteration + 1}")
        iteration_start_time = time.time()

        async def validate_question(question):
            start_time = time.time()
            try:
                rule_based_result = await rule_based_validate(question, questions, checklist)
                llm_based_result = await llm_based_validate(question, rule_based_result, checklist)
                meta_result = await meta_validate(question, rule_based_result, llm_based_result)
                end_time = time.time()
                logger.info(f"Validated question UUID {question['uuid']} in {end_time - start_time:.2f} seconds")
                return {
                    "is_valid": meta_result["is_valid"],
                    "data": meta_result["data"],
                    "checklist_source": checklist_source,
                    "iteration": iteration + 1
                }
            except Exception as e:
                end_time = time.time()
                logger.error(f"Validation failed for UUID {question.get('uuid', 'unknown')}: {e}")
                logger.info(f"Validation attempt for UUID {question.get('uuid', 'unknown')} completed in {end_time - start_time:.2f} seconds")
                return {
                    "is_valid": False,
                    "data": {
                        "is_valid": False,
                        "validation_score": 1,
                        "errors": [f"Validation failed: {str(e)}"],
                        "warnings": [],
                        "feedback": "Validation failed due to error.",
                        "improvement_recommendations": {"specific_fixes": []},
                        "final_decision": "REJECT"
                    },
                    "checklist_source": checklist_source,
                    "iteration": iteration + 1
                }

        tasks = [validate_question(q) for q in questions]
        iteration_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend([r for r in iteration_results if isinstance(r, dict)])

        iteration_end_time = time.time()
        logger.info(f"Validation iteration {iteration + 1} completed in {iteration_end_time - iteration_start_time:.2f} seconds")

    end_time = time.time()
    logger.info(f"validate_questions completed in {end_time - start_time:.2f} seconds")
    return results

@app.post("/validate_excel")
async def validate_excel(file: UploadFile = File(...)):
    """Validate questions from Excel file."""
    start_time = time.time()
    try:
        questions = await parse_excel(file)
        results = await validate_questions(questions)
        final_results = [r["data"] for r in results if r["iteration"] == 2]
        end_time = time.time()
        logger.info(f"validate_excel completed in {end_time - start_time:.2f} seconds")
        return JSONResponse(content={"results": final_results})
    except Exception as e:
        end_time = time.time()
        logger.error(f"validate_excel failed: {e}")
        logger.info(f"validate_excel failed in {end_time - start_time:.2f} seconds")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

@app.post("/validate_json")
async def validate_json(request: Request):
    """Validate questions from JSON data."""
    start_time = time.time()
    try:
        json_data = await request.json()
        questions = await parse_json(json_data)
        results = await validate_questions(questions)
        final_results = [r["data"] for r in results if r["iteration"] == 2]
        end_time = time.time()
        logger.info(f"validate_json completed in {end_time - start_time:.2f} seconds")
        return JSONResponse(content={"results": final_results})
    except json.JSONDecodeError as e:
        end_time = time.time()
        logger.error(f"Invalid JSON: {e}")
        logger.info(f"validate_json failed in {end_time - start_time:.2f} seconds")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        end_time = time.time()
        logger.error(f"validate_json failed: {e}")
        logger.info(f"validate_json failed in {end_time - start_time:.2f} seconds")
        raise HTTPException(status_code=500, detail=f"Failed to validate: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)