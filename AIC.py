import pandas as pd
import os
import json
import asyncio
import logging
import argparse
import re
from datetime import datetime
from urllib.parse import quote
from typing import List, Dict, Any, Type
import tenacity
from openai import RateLimitError
from tqdm.asyncio import tqdm_asyncio
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_openai import ChatOpenAI
from dateutil.parser import parse as date_parse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from web_page_retrieval import scrape_google_search_results
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import hashlib
from tqdm import tqdm
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from metrics import ComprehensiveMetrics


"""
example usage:
python openai_modules/generalized/general_eval.py \
  --input_file "openai_modules/generalized/data.csv" \
  --output_file "results.csv" \
  --dataset_description "A list of organizations in the United States." \
  --schema_fields name city state
"""

class ComprehensiveFormatter(logging.Formatter):
    def format(self, record):
        details = getattr(record, 'details', None)
        if not details:
            return super().format(record)

        decision = details.get('final_decision', 'N/A')
        reason = details.get('final_reasoning', 'No reasoning provided')

        if decision == 'N/A':
            if 'arbiter_decision' in details and details['arbiter_decision']:
                if isinstance(details['arbiter_decision'], dict):
                    decision = details['arbiter_decision'].get('decision', 'N/A')
                    reason = details['arbiter_decision'].get('reasoning', '')
                else:
                    decision = getattr(details['arbiter_decision'], 'decision', 'N/A')
                    reason = getattr(details['arbiter_decision'], 'reasoning', '')
            elif 'rejection_reason' in details:
                decision = "REJECT"
                reason = details['rejection_reason']

        source_url = "No source URL"
        row_id = "Unknown"
        if 'original_data' in details and details['original_data']:
            source_url = details['original_data'].get('source', 'No source URL')
        if 'row_id' in details:
            row_id = details['row_id']

        timestamp = self.formatTime(record)

        summary_lines = []
        if 'relevancy_report' in details and details['relevancy_report']:
            summary_lines.append(f"Relevancy: {'OK' if details['relevancy_report'].get('is_relevant') else 'FAIL'}")
        if 'layout_analysis_report' in details and details['layout_analysis_report']:
            summary_lines.append(f"Layout: {details['layout_analysis_report'].get('layout_type', 'Unknown')}")
        if 'content_extraction_report' in details and details['content_extraction_report']:
            summary_lines.append(f"Extraction: {'OK' if details['content_extraction_report'].get('cleaned_text') else 'FAIL'}")
        if 'source_scrutinizer_report' in details and details['source_scrutinizer_report']:
            summary_lines.append(f"Source: {details['source_scrutinizer_report'].get('reliability', 'Unknown')}")
        if 'fact_checker_report' in details and details['fact_checker_report']:
            fc_report = details['fact_checker_report']
            summary_lines.append(f"Fact Check: {'OK' if fc_report.get('is_article_content') and fc_report.get('is_accurate') else 'FAIL'}")
        if 'remediation_report_same' in details and details['remediation_report_same']:
            summary_lines.append(f"Remediation (Same): {details['remediation_report_same'].get('status', 'Unknown')}")
        if 'remediation_report_different' in details and details['remediation_report_different']:
            summary_lines.append(f"Remediation (Different): {details['remediation_report_different'].get('status', 'Unknown')}")


        summary = " | ".join(summary_lines) if summary_lines else "No agent reports available"

        header = (
            f"\n======================================================================\n"
            f"[{timestamp}] | ROW: {row_id} | FINAL DECISION: {decision}\n"
            f"SOURCE: {source_url}\n"
            f"REASON: {reason if reason else 'No reason provided'}\n"
            f"SUMMARY: {summary}\n"
            f"======================================================================\n"
        )
        body = json.dumps(details, indent=4)
        return header + body

def setup_logger(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("AICommitteeLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(ComprehensiveFormatter(fmt='%(asctime)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger

def hierarchical_deduplication(accepted_results: list, schema_fields: list) -> list:
    if not accepted_results:
        return []

    df = pd.DataFrame(accepted_results)
    for field in schema_fields:
        df = df[df[field].notna() & (df[field] != "")]

    if 'date' in schema_fields and 'date' in df.columns:
        non_date_fields = [col for col in schema_fields if col != 'date']
        seen_full_date = set()
        seen_year_month = set()
        seen_year_only = set()
        final_rows = []

        for _, row in df.iterrows():
            base_fingerprint = tuple(row[field] for field in non_date_fields)

            date_str = str(row.get('date', ''))
            parts = date_str.split('-')

            year, month, day = None, None, None

            try:
                if len(parts) >= 1 and parts[0].isdigit():
                    year = int(parts[0])
                if len(parts) >= 2 and parts[1].isdigit():
                    month = int(parts[1])
                if len(parts) == 3 and parts[2].isdigit():
                    day = int(parts[2])
            except (ValueError, IndexError):
                continue

            if day is not None and month is not None and year is not None:
                fingerprint = base_fingerprint + (year, month, day)
                if fingerprint not in seen_full_date:
                    seen_full_date.add(fingerprint)
                    final_rows.append(row)
            elif month is not None and year is not None:
                fingerprint = base_fingerprint + (year, month)
                if fingerprint not in seen_year_month:
                    seen_year_month.add(fingerprint)
                    final_rows.append(row)
            elif year is not None:
                fingerprint = base_fingerprint + (year,)
                if fingerprint not in seen_year_only:
                    seen_year_only.add(fingerprint)
                    final_rows.append(row)

        if not final_rows:
            return []
        final_df = pd.DataFrame(final_rows)
        return final_df.to_dict('records')
    else:
        df = df.drop_duplicates(subset=schema_fields)
        return df.to_dict('records')

class LayoutAnalysisResponse(BaseModel):
    layout_type: str = Field(..., description="Classification of the page layout (e.g., 'ARTICLE', 'DIRECTORY_LISTING', 'TAG_PAGE', 'SEARCH_RESULTS', 'HOMEPAGE', 'ERROR_PAGE', 'OTHER').")
    reasoning: str = Field(..., description="Brief justification for the layout classification based on structural elements.")

class ContentExtractionResponse(BaseModel):
    cleaned_text: str | None = Field(None, description="The extracted raw text of the main content block, stripped of navigation, headers, and footers. Null if no main content block could be confidently identified.")

class FactCheckerResponse(BaseModel):
    is_article_content: bool = Field(..., description="True if the provided text contains meaningful information related to the data point, even if partially parsed or formatted. Only false if completely empty, pure error messages, or completely unrelated content.")
    is_accurate: bool | None = Field(None, description="True if the data point is supported by the text. Set to None if is_article_content is false.")
    extracted_date: str | None = Field(None, description="The date from the text. Use the most specific format found (e.g., YYYY-MM-DD, YYYY-MM, or YYYY). Do not add missing days or months.")
    confidence: float | None = Field(None, description="Confidence score for the accuracy assessment. Set to None if is_article_content is false.")
    notes: str = Field(..., description="If not article content, explain why. Otherwise, explain the fact-checking reasoning for each field in the data point.")

class SourceScrutinizerResponse(BaseModel):
    source_type: str = Field(..., description="Classification of the source (e.g., 'Major News Outlet', 'Local News', 'Government', 'University', 'Press Release', 'Blog', 'Unknown').")
    reliability: str = Field(..., description="Reliability assessment ('Very High', 'High', 'Moderate', 'Low', 'Very Low').")
    notes: str = Field(..., description="Brief justification for the classification and reliability assessment.")


class ArbiterResponse(BaseModel):
    decision: str = Field(..., description="The final decision, either 'ACCEPT' or 'REJECT'.")
    reasoning: str = Field(..., description="A clear explanation for the decision, referencing the specific criteria met.")

class RelevancyResponse(BaseModel):
    is_relevant: bool = Field(..., description="True if the data point is topically relevant to the dataset description.")
    reasoning: str = Field(..., description="A brief explanation for the relevancy decision.")

class RemediationSameResponse(BaseModel):
    status: str = Field(..., description="The outcome of the remediation attempt ('SUCCESS', 'FAILED').")
    updated_data: Dict | None = Field(None, description="A dictionary containing ONLY the key-value pairs that were fixed or newly found. None if status is 'FAILED'.")
    reasoning: str = Field(..., description="A step-by-step explanation of how the data was found or calculated. If it failed, explains why.")

class UnifiedRemediationResponse(BaseModel):
    remediation_result: RemediationSameResponse | None = Field(None, description="The completed remediation object if the data could be fixed directly from the text.")
    tool_request: dict | None = Field(None, description="A JSON object for a tool call if more information is needed for a calculation.")

class RemediationDifferentResponse(BaseModel):
    status: str = Field(..., description="The outcome of the discovery attempt ('SUCCESS', 'FAILED').")
    found_data: List[Dict] = Field(..., description="A list of new data point dictionaries discovered on the page. Empty if none were found.")
    reasoning: str = Field(..., description="A summary of what was found or why the search failed.")

class DataIntegrityResponse(BaseModel):
    is_plausible: bool = Field(..., description="True if the data point is plausible and does not contain obvious errors, False otherwise.")
    reasoning: str = Field(..., description="A brief explanation for the decision, highlighting any specific fields that are implausible.")

class RemediationAnalystResponse(BaseModel):
    analysis: str = Field(..., description="A plain-text, step-by-step analysis of how to fix the data point.")
    is_direct_fix_possible: bool = Field(..., description="True if the Source Text contains the explicit answer.")
    is_calculation_needed: bool = Field(..., description="True if a calculation is required using a relative value from the text.")
    question_for_tool: str | None = Field(None, description="If a calculation is needed, the specific question to ask the fact-lookup tool.")
    direct_answer: str | None = Field(None, description="If a direct fix is possible, the explicit value found in the text.")

class ContextLearningResponse(BaseModel):
    entity_patterns: Dict[str, str] = Field(..., description="Maps each field to its expected entity type and characteristics")
    temporal_context: str = Field(..., description="Description of how dates/times should be interpreted in this dataset")
    extraction_guidelines: List[str] = Field(..., description="Specific rules for extraction based on the dataset analysis")
    negative_examples: Dict[str, List[str]] = Field(..., description="Common wrong extractions to avoid for each field")

class FallacyExample(BaseModel):
    good_example: str = Field(..., description="A clear example of a correct semantic match for this dataset, including the source text snippet and the extracted value.")
    bad_example: str = Field(..., description="A clear example of a common fallacy or incorrect semantic match for this dataset, including the source text snippet and the extracted value.")
    explanation: str = Field(..., description="A brief explanation of why the bad example is wrong and the good one is right, referencing the specific principle.")

class ContextualExamplesResponse(BaseModel):
    examples: Dict[str, FallacyExample] = Field(..., description="A dictionary where keys are fallacy types (e.g., 'Whole Entity Principle', 'Scope Mismatch') and values are the corresponding examples.")

class MetaContextAgent:
    """Combined agent that handles both context learning and contextual example generation."""

    def __init__(self, llm_caller):
        self.llm_caller = llm_caller
        self.learned_context = None
        self.contextual_examples = None

    async def initialize(self, dataset_description: str, schema_fields: List[str], input_df: pd.DataFrame = None):
        """Initialize the meta context agent by learning from dataset and generating examples."""

        if input_df is not None and len(input_df) > 0:
            await self._learn_from_dataset(input_df, dataset_description, schema_fields)

        self.contextual_examples = await self._generate_examples(dataset_description, schema_fields)

    async def _learn_from_dataset(self, df: pd.DataFrame, dataset_description: str, schema_fields: List[str]):
        """Learn context patterns from sample dataset rows."""

        sample_size = min(10, len(df))
        sample_rows = df.head(sample_size).to_dict('records')

        analysis_prompt = f"""
You are a data analysis expert. Analyze these sample rows from a dataset to understand what types of entities should be extracted for each field.

**Dataset Description:** "{dataset_description}"

**Schema Fields:** {schema_fields}

**Sample Rows:**
{json.dumps(sample_rows, indent=2)}

Based on these samples, determine:

1. **Entity Patterns**: For each field, what type of entity should it contain? (e.g., "geographic regions at state level", "organizational names", "specific dates when events occurred")

2. **Temporal Context**: If there are date fields, what do the dates represent? (event dates, publication dates, reporting periods, etc.)

3. **Extraction Guidelines**: What specific rules should guide extraction? Focus on entity granularity and common pitfalls.

4. **Negative Examples**: For each field, what are examples of WRONG extractions that should be avoided? Think about partial entities, wrong granularity, scope mismatches, etc.

**CRITICAL: Analyze the geographic/organizational scope carefully**
- If this appears to be national-level data, state/city/organization-specific data should be REJECTED
- If this appears to be state-level data, city/organization-specific data should be REJECTED
- If this appears to be organization-level data, broader geographic data should be REJECTED
- Look for scope mismatches where data at one level is incorrectly assigned to a different level

Be very specific about the expected granularity and type for each field based on the patterns you see in the sample data.
"""

        try:
            self.learned_context = await self.llm_caller(analysis_prompt, ContextLearningResponse)
            if not self.learned_context:
                self.learned_context = self._generate_fallback_context(schema_fields)
        except Exception as e:
            self.learned_context = self._generate_fallback_context(schema_fields)

    async def _generate_examples(self, dataset_description: str, schema_fields: List[str]) -> ContextualExamplesResponse | None:
        """Generate contextual examples for fallacies."""

        prompt = f"""
You are an expert in data validation and logical fallacies. Your task is to create illustrative examples of common semantic fallacies based on a specific dataset context. These examples will be used to train another AI to be a better fact-checker.

**Dataset Context:**
- **Description:** "{dataset_description}"
- **Schema Fields:** {json.dumps(schema_fields)}

For each of the three logical fallacies listed below, please create one **good example** (a correct semantic match) and one **bad example** (an incorrect match that commits the fallacy). The examples must be highly specific and relevant to the provided dataset context.

**Fallacies to Illustrate:**

1.  **The Whole Entity Principle:**
    - The bad example should demonstrate extracting a value that is merely a fragment of a larger proper noun, where that proper noun represents a different kind of entity than what the schema field requires.
    - *Example Goal:* If the schema needs a `city`, a bad example would be extracting 'Boston' from the text 'the Boston Globe newspaper', because the primary entity is a newspaper, not the city.

2.  **The Scope Mismatch Principle:**
    - The bad example should illustrate finding data that is at the wrong level of granularity or scope compared to what the dataset requires.
    - *Example Goal:* If the dataset is for state-level statistics, a bad example would be using a statistic that is explicitly about a single city/organization within that state.

3.  **The Semantic Equivalence Principle:**
    - The bad example should show a concept that is related to the required value but is fundamentally different in meaning.
    - *Example Goal:* If the schema requires `crime_type: 'Fraud'`, a bad example would be matching it to text describing 'budget cuts', as they are not semantically equivalent.

For each fallacy, provide a good example, a bad example, and a clear, concise explanation.
"""
        try:
            return await self.llm_caller(prompt, ContextualExamplesResponse)
        except Exception as e:
            return None

    def _generate_fallback_context(self, schema_fields):
        """Generate basic context when learning fails"""
        return ContextLearningResponse(
            entity_patterns={field: f"Complete {field} entities" for field in schema_fields},
            temporal_context="Dates should represent when the described event occurred",
            extraction_guidelines=["Extract complete entity names", "Avoid partial matches"],
            negative_examples={field: [f"partial {field} names"] for field in schema_fields}
        )

    def generate_context_rules(self, target_field: str = None) -> str:
        """Generate context-aware extraction rules for prompts"""
        if not self.learned_context:
            return ""

        rules = [
            "**DATASET-SPECIFIC CONTEXT RULES:**",
            f"Temporal Context: {self.learned_context.temporal_context}",
            "",
            "**Entity Type Expectations:**"
        ]

        for field, pattern in self.learned_context.entity_patterns.items():
            rules.append(f"- {field}: {pattern}")

        rules.extend([
            "",
            "**CRITICAL SCOPE VALIDATION:**",
            "- REJECT data if it represents a different scope/granularity than expected",
            "- Campus/organization data ≠ State/national data",
            "- City data ≠ State data",
            "- Subset data ≠ Total data",
            "",
            "**Extraction Guidelines:**"
        ])
        rules.extend([f"- {guideline}" for guideline in self.learned_context.extraction_guidelines])

        if target_field and target_field in self.learned_context.negative_examples:
            rules.extend([
                f"",
                f"**AVOID these common mistakes for '{target_field}':**"
            ])
            rules.extend([f"- {example}" for example in self.learned_context.negative_examples[target_field]])

        return "\n".join(rules) + "\n\n"

class APIMetrics:
    def __init__(self, price_input, price_output):
        self.total_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.price_per_input_token = price_input
        self.price_per_output_token = price_output

    def update(self, response):
        self.total_requests += 1
        token_usage = getattr(response, 'response_metadata', {}).get("token_usage", {})
        self.total_prompt_tokens += token_usage.get("prompt_tokens", 0)
        self.total_completion_tokens += token_usage.get("completion_tokens", 0)

    def log_failure(self):
        self.failed_requests += 1

    def report(self):
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        input_cost = self.total_prompt_tokens * self.price_per_input_token
        output_cost = self.total_completion_tokens * self.price_per_output_token
        total_cost = input_cost + output_cost
        print(f"api_usage success={self.total_requests} failed={self.failed_requests} tokens={total_tokens} cost=${total_cost:.4f}")

class ContentManager:
    def __init__(self, cache_dir: str = "./markdown_cache", logger=None, metrics_tracker=None):
        self.cache_dir = cache_dir
        self.logger = logger
        self.metrics_tracker = metrics_tracker
        self.crawler = AsyncWebCrawler()
        self._file_lock = asyncio.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
        self.crawl_successes = 0
        self.crawl_failures = 0

    def _get_cache_path(self, url: str) -> str:
        safe_filename = quote(url, safe='')
        if len(safe_filename) > 200:
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            safe_filename = f"{safe_filename[:180]}_{url_hash}"
        return os.path.join(self.cache_dir, f"{safe_filename}.md")

    async def get_content(self, url: str) -> str | None:
        cache_path = self._get_cache_path(url)

        if os.path.exists(cache_path):
            if self.logger:
                self.logger.info(f"Cache HIT for URL: {url}")
            self.cache_hits += 1
            content = None
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if self.metrics_tracker:
                # Estimate content quality based on length and structure
                quality_score = min(1.0, len(content) / 5000.0) if content else 0.0
                self.metrics_tracker.track_web_scraping(url, success=True, cached=True, content_quality=quality_score)
            return content

        if self.logger:
            self.logger.info(f"Cache MISS. Crawling URL: {url}")
        self.cache_misses += 1
        
        config = CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator())
        result = await self.crawler.arun(url, config=config)

        if not result.success or not result.markdown:
            if self.logger:
                self.logger.error(f"Crawl failed for {url}", extra={'error': result.error_message})
            self.crawl_failures += 1
            if self.metrics_tracker:
                self.metrics_tracker.track_web_scraping(url, success=False, cached=False, content_quality=0.0)
            return None
        self.crawl_successes += 1

        markdown = re.sub(r'\n{3,}', '\n\n', result.markdown).strip()

        async with self._file_lock:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                if self.logger:
                    self.logger.info(f"Saved content for {url} to {cache_path}")
            except IOError as e:
                if self.logger:
                    self.logger.error(f"Failed to write to cache file {cache_path}", extra={'error': str(e)})

        if self.metrics_tracker:
            # Estimate content quality based on length and structure
            quality_score = min(1.0, len(markdown) / 5000.0) if markdown else 0.0
            self.metrics_tracker.track_web_scraping(url, success=True, cached=False, content_quality=quality_score)

        return markdown
    
    async def close(self):
        await self.crawler.close()

class AICommittee:
    def __init__(self, llm_model_name: str, metrics_tracker: ComprehensiveMetrics, logger=None):
        self.model_name = llm_model_name
        self.llm = self._initialize_llm_with_fallback(llm_model_name)
        self.metrics_tracker = metrics_tracker
        self.logger = logger
        self.DataModel: Type[BaseModel] = None
        self.search_templates: List[str] = []
        self.search_semaphore = asyncio.Semaphore(1)
        self.content_manager = ContentManager(logger=self.logger, metrics_tracker=metrics_tracker)
        self.meta_context_agent = MetaContextAgent(self._call_llm_with_structure)
        self._current_row_id: str | None = None
        self._current_agent_name: str | None = None

    def set_metrics_context(self, row_id: str | None, agent_name: str | None):
        self._current_row_id = row_id
        self._current_agent_name = agent_name

    def _supports_temperature(self, model_name: str) -> bool:
        """Check if a model supports temperature parameter"""
        models_no_temp_control = [
            'o1', 'o1-mini', 'o1-preview', 'o1-pro',
            'o3', 'o3-mini', 'o3-pro', 'o3-deep-research',
            'o4-mini', 'o4-mini-deep-research',
            'text-davinci-003', 'text-davinci-002', 'text-curie-001',
            'text-babbage-001', 'text-ada-001',
        ]
        return model_name not in models_no_temp_control

    def _initialize_llm_with_fallback(self, model_name: str):
        """Initialize LLM with parameter fallback for different model types"""

        is_restricted = not self._supports_temperature(model_name)

        try:
            if is_restricted:
                print(f"Model '{model_name}' doesn't support temperature/seed control. Using default parameters.")
                class RestrictedChatOpenAI(ChatOpenAI):
                    @property
                    def _default_params(self):
                        params = super()._default_params
                        params.pop('temperature', None)
                        params.pop('seed', None)
                        return params

                return RestrictedChatOpenAI(model=model_name)
            else:
                print(f"Initializing {model_name} with temperature=0.0 for consistent outputs.")
                return ChatOpenAI(model=model_name, temperature=0.0)
        except Exception as e:

            try:
                if not self._supports_temperature(model_name):
                    class RestrictedChatOpenAI(ChatOpenAI):
                        @property
                        def _default_params(self):
                            params = super()._default_params
                            params.pop('temperature', None)
                            params.pop('seed', None)
                            return params
                    return RestrictedChatOpenAI(model=model_name)
                else:
                    return ChatOpenAI(model=model_name)
            except Exception as e2:

                return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    async def _initialize_framework(self, dataset_description: str, schema_fields: List[str], input_df: pd.DataFrame = None):
        self.dataset_description = dataset_description

        await self.meta_context_agent.initialize(dataset_description, schema_fields, input_df)

        self.DataModel = await self._generate_pydantic_schema(dataset_description, schema_fields)
        self.search_templates = await self._generate_search_templates(dataset_description, schema_fields)



    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(RateLimitError), before_sleep=lambda retry_state: print(f"Rate limit hit. Retrying in {retry_state.next_action.sleep:.2f} seconds..."))
    async def _call_llm(self, prompt: str) -> str:
        try:
            message = await self.llm.ainvoke(prompt)
            try:
                self.metrics_tracker.update(message, agent_name=self._current_agent_name, row_id=self._current_row_id)
            except Exception:
                pass
            return message.content
        except Exception as e:
            if ("temperature" in str(e) and ("not support" in str(e) or "unsupported" in str(e).lower())) or \
               ("seed" in str(e) and ("not support" in str(e) or "unsupported" in str(e).lower())):

                class RestrictedChatOpenAI(ChatOpenAI):
                    @property
                    def _default_params(self):
                        params = super()._default_params
                        params.pop('temperature', None)
                        params.pop('seed', None)
                        return params
                self.llm = RestrictedChatOpenAI(model=self.model_name)
                message = await self.llm.ainvoke(prompt)
                try:
                    self.metrics_tracker.update(message, agent_name=self._current_agent_name, row_id=self._current_row_id)
                except Exception:
                    pass
                return message.content
            else:
                try:
                    if self._current_agent_name:
                        self.metrics_tracker.record_retry(self._current_agent_name)
                    self.metrics_tracker.log_failure(row_id=self._current_row_id, agent_name=self._current_agent_name)
                except Exception:
                    pass
                if self.logger:
                    self.logger.error("LLM call failed", extra={'error': str(e)})
                raise

    async def _call_llm_with_structure(self, prompt: str, structure: BaseModel) -> BaseModel | None:
        model_schema = structure.schema()
        json_prompt = (
            f"{prompt}\n\n"
            f"Please provide your response as a single JSON object that strictly adheres to the following Pydantic schema. "
            f"Do not include any other text or explanations outside of the JSON object.\n"
            f"Schema:\n{json.dumps(model_schema, indent=2)}"
        )
        try:
            message = await self.llm.ainvoke(json_prompt)
            try:
                self.metrics_tracker.update(message, agent_name=self._current_agent_name, row_id=self._current_row_id)
            except Exception:
                pass
            json_str = message.content
            if '```json' in json_str:
                json_str = json_str.split('```json\n', 1)[1].split('\n```', 1)[0]
            validated_response = structure.parse_raw(json_str)
            return validated_response
        except RateLimitError as e:
            try:
                if self._current_agent_name:
                    self.metrics_tracker.record_retry(self._current_agent_name)
                self.metrics_tracker.log_failure(row_id=self._current_row_id, agent_name=self._current_agent_name)
            except Exception:
                pass
            if self.logger:
                self.logger.error("Rate limit error after retries", extra={'error': str(e)})
            raise
        except Exception as e:
            if ("temperature" in str(e) and ("not support" in str(e) or "unsupported" in str(e).lower())) or \
               ("seed" in str(e) and ("not support" in str(e) or "unsupported" in str(e).lower())):

                class RestrictedChatOpenAI(ChatOpenAI):
                    @property
                    def _default_params(self):
                        params = super()._default_params
                        params.pop('temperature', None)
                        params.pop('seed', None)
                        return params
                self.llm = RestrictedChatOpenAI(model=self.model_name)
                message = await self.llm.ainvoke(json_prompt)
                try:
                    self.metrics_tracker.update(message, agent_name=self._current_agent_name, row_id=self._current_row_id)
                except Exception:
                    pass
                json_str = message.content
                if '```json' in json_str:
                    json_str = json_str.split('```json\n', 1)[1].split('\n```', 1)[0]
                validated_response = structure.parse_raw(json_str)
                return validated_response
            else:
                try:
                    if self._current_agent_name:
                        self.metrics_tracker.record_retry(self._current_agent_name)
                    self.metrics_tracker.log_failure(row_id=self._current_row_id, agent_name=self._current_agent_name)
                except Exception:
                    pass
                raw_content = message.content if 'message' in locals() else "No content received"
                if self.logger:
                    self.logger.error("Failed to parse LLM response", extra={'structure': structure.__name__, 'error': str(e), 'raw_content': raw_content})
                return None

    async def _generate_pydantic_schema(self, data_description: str, schema_fields: List[str]) -> Type[BaseModel]:
        print("programmatically generating pydantic schema")
        try:
            class_name = re.sub(r'[^a-zA-Z0-9]', '', data_description.title()) or "DynamicDataModel"
            pydantic_fields = {}
            for field_name in schema_fields:
                field_type = Any
                description = f"The {field_name.replace('_', ' ')}."
                if "date" in field_name.lower():
                    description = f"The {field_name.replace('_', ' ')}. Use the most specific format found (e.g., YYYY-MM-DD, YYYY-MM, or YYYY). Do not add default values for missing months or days."
                pydantic_fields[field_name] = (field_type, Field(..., description=description))
            DynamicModel = create_model(class_name, **pydantic_fields)
            print(f"Successfully created dynamic schema: {class_name}")
            return DynamicModel
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to create dynamic Pydantic model", extra={'error': str(e)})
            return self._create_fallback_schema(schema_fields)

    def _create_fallback_schema(self, schema_fields: List[str]) -> Type[BaseModel]:
        field_definitions = {}
        for field in schema_fields:
            field_definitions[field] = (Any, Field(..., description=f"The {field} field"))
        return type('FallbackDataModel', (BaseModel,), field_definitions)

    async def _generate_search_templates(self, data_description: str, placeholder_names: List[str]) -> List[str]:
        prompt = f"""
Generate 5 Google Search query templates for the following dataset description: {data_description}
Important: Focus on creating concise queries that will find individual data points or small pieces of information, NOT complete datasets. Each query should aim to find pages that contain just one or a few relevant pieces of information. Make sure to include important keywords in the query.
Each template should be extremely short and straightforward. Mostly reuse words from the dataset description. The templates should contain the following placeholder variables, surrounded by curly braces:
{json.dumps(placeholder_names)}
Return ONLY a JSON array of template strings. Do not include any explanation or additional text.
"""
        response_str = await self._call_llm(prompt)
        if '```json' in response_str:
            response_str = response_str.split('```json\n', 1)[1].split('\n```', 1)[0]
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            if self.logger:
                self.logger.error("Failed to decode JSON from search template generation", extra={'response': response_str})
            return []

    async def layout_analyzer_agent(self, page_markdown: str):
        self.metrics_tracker.start_agent("layout_analyzer")
        prompt = f"""
You are a web page structural analyst. Your task is to analyze the structural layout of the provided markdown text and classify it.
Focus *only* on the structure (headings, lists, text length, repeating patterns) and ignore the actual content or topic.

**Classification Guide:**
- **ARTICLE**: Long-form text, headings, paragraphs. A typical news article or blog post.
- **DIRECTORY_LISTING**: A primary list of items (people, businesses, data entries) with brief, structured details for each. Often, each item is a link. The main purpose of the page is this list.
- **TAG_PAGE / SEARCH_RESULTS**: The page content is a list of links to other articles that share a tag or search query. Usually consists of headlines and short snippets.
- **HOMEPAGE**: A top-level page with a mix of many different elements, promotional content, and broad navigation to the rest of the site.
- **ERROR_PAGE**: A page indicating an error (e.g., "404 Not Found", "Access Denied").
- **OTHER**: Use this for any other layout that doesn't fit the above categories (e.g., a login form, a pure multimedia gallery).

Analyze the following markdown and classify its layout.

**Markdown Text:**
"{page_markdown[:16000]}"
"""
        try:
            return await self._call_llm_with_structure(prompt, LayoutAnalysisResponse)
        finally:
            self.metrics_tracker.end_agent("layout_analyzer")

    async def content_extractor_agent(self, page_markdown: str):
        self.metrics_tracker.start_agent("content_extractor")
        prompt = f"""
This page was flagged as navigational, but this may be an error. Ignore all headers, footers, sidebars, and navigational templates.
Your only task is to find the single largest block of main content (like an article body or a list of items) and extract its raw text.
If no true article body or main content block exists, return null for the 'cleaned_text' field.
"""
        try:
            return await self._call_llm_with_structure(prompt, ContentExtractionResponse)
        finally:
            self.metrics_tracker.end_agent("content_extractor")

    def _format_contextual_examples(self) -> str:
        """Helper function to format the generated contextual examples for the prompt."""
        if not self.meta_context_agent.contextual_examples or not self.meta_context_agent.contextual_examples.examples:
            return ""

        lines = ["\n**DYNAMIC CONTEXTUAL EXAMPLES (Generated for this specific dataset)**"]
        for fallacy, example in self.meta_context_agent.contextual_examples.examples.items():
            lines.append(f"\n**Fallacy Type: {fallacy}**")
            lines.append(f"- **Correct Match Example:** {example.good_example}")
            lines.append(f"- **Incorrect Match (Fallacy) Example:** {example.bad_example}")
            lines.append(f"- **Explanation:** {example.explanation}")

        return "\n".join(lines) + "\n"

    async def fact_checker_agent_from_content(self, data_point: Dict[str, Any], markdown: str, layout_report: LayoutAnalysisResponse):
        self.metrics_tracker.start_agent("fact_checker")
        analysis_hint = ""
        if layout_report:
            layout_type = layout_report.layout_type
            if layout_type in ['DIRECTORY_LISTING', 'TAG_PAGE', 'SEARCH_RESULTS']:
                analysis_hint = (
                    f"**Analysis Hint:** The page layout has been identified as '{layout_type}'. "
                    f"This means the content is likely a structured list where each item (often a hyperlink) represents a data point. "
                    f"Do NOT dismiss this page as purely navigational. Your primary goal is to find data within the list items, headlines, and snippets. "
                    f"Evaluate the text within and around the links as valid content."
                )
            elif layout_type == 'ARTICLE':
                analysis_hint = "**Analysis Hint:** The page layout has been identified as 'ARTICLE'. Look for facts within the main body of the text, such as paragraphs and sections."
            elif layout_type == 'ERROR_PAGE':
                 analysis_hint = "**Analysis Hint:** The page layout has been identified as 'ERROR_PAGE'. The content is likely an error message and should not contain the data point."
            elif layout_type == 'HOMEPAGE':
                 analysis_hint = "**Analysis Hint:** The page layout has been identified as 'HOMEPAGE'. While it may contain a lot of navigation, scan carefully for data-rich sections like 'Latest News' or featured items that might contain the data point."

        context_rules = self.meta_context_agent.generate_context_rules()
        contextual_examples_str = self._format_contextual_examples()

        prompt = (
            f"You are a meticulous fact-checker. Your task is to determine if the Data Point provided is factually supported by the Source Text.\n\n"
            f"{analysis_hint}\n\n"
            f"{context_rules}"

            f"**GUIDING PRINCIPLES OF INTERPRETATION**\n"
            f"You must apply the following common-sense rules in ALL OF YOUR ANALYSES:\n"
            f"1.  **Numbers vs. Words:** Numbers can be written as digits (e.g., `100`) or as words (e.g., 'one hundred'). You must treat them as identical.\n"
            f"2.  **Meaning vs. Keywords:** Focus on the intended meaning of a field, not just the exact word. A data point is valid if the source text uses a different word but describes the same core concept. For example, a data point requiring a `genre` of 'Fiction' is directly supported if the text calls the book 'a novel'.\n"
            f"3.  **Context is Key:** Use the document's structure and topic to understand implied facts. Information in a section titled 'Q3 Financial Results' can be assumed to be about the third quarter, even if 'Q3' isn't repeated in every sentence. If an article is about a specific individual, a mention of 'his work' implies that individual as the actor.\n\n"

            f"**CRITICAL SEMANTIC AUDIT: BEYOND LITERAL MATCHING**\n"
            f"{contextual_examples_str}\n"
            f"Before accepting any value, you MUST perform a deeper semantic check based on the principles below, using the dynamic examples above as a guide for this specific dataset. A value can be literally present in the text but contextually wrong. You are required to check for the following three fallacies:\n"
            f"1.  **The Whole Entity Principle:** You must verify that the extracted value represents the *complete and correct entity type*, not just a fragment of a different one. Ask yourself: Is this value merely a component of a larger proper noun that is a different kind of thing? For example, if the schema demands a `location` and the text mentions 'the Boston Globe newspaper,' extracting 'Boston' would be an error. The context is about a *newspaper organization*, not the geographical city itself. You must reject such fragmented, out-of-context matches.\n"
            f"2.  **The Scope Mismatch Principle:** You must verify that the data represents the correct scope/granularity. State data ≠ Country data (unless the state is the only one participating). Organization data ≠ Geographic region data. The data scope must exactly match the expected entity scope.\n"
            f"3.  **The Qualifier Principle (Revised):** You must analyze words that modify a potential data point. Your goal is to distinguish between reasonable estimations and modifiers that reverse or mischaracterize the meaning.\n"
            f"    - **ACCEPTABLE Qualifiers:** Words indicating reasonable estimation (e.g., 'approximately', 'about', 'around') or setting a boundary (e.g., 'more than 100', 'fewer than 50') are generally acceptable, as they still ground the data point in a factual, numeric range.\n"
            f"    - **UNACCEPTABLE Qualifiers:** You MUST REJECT the data if a qualifier fundamentally changes the meaning. Be vigilant for:\n"
            f"        - **Conditionality, Negation, or Incomplete Status:** Words that cast doubt on the core fact (e.g., '*potential* outcome', '*planned* release', '*failed* to reach', '*almost* finished'). These change the assertion from a fact that *has occurred* to one that *might* or *has not* occurred.\n"
            f"        - **Temporal Mischaracterization:** You must evaluate whether temporal expressions appropriately describe when an event occurred versus describing durations or ongoing states.\n"
            f"            * **ACCEPTABLE Date Ranges:** Date ranges that specify when an event occurred are acceptable. Examples: 'the theft happened between January and March 2020', 'the construction project ran from March to May 2019', 'employed from 2018-2021', 'the breach was discovered between June 15-20, 2019'.\n"
            f"            * **UNACCEPTABLE Temporal Expressions:** Reject only **recurring events** (e.g., 'meetings occur every Tuesday', 'happens annually', 'monthly reports') and **ongoing indefinite states** (e.g., 'active since 2020' with no end). These describe patterns or open-ended states, not specific events or time-bounded activities.\n"
            f"4.  **The Semantic Equivalence Principle:** You must evaluate whether the source text supports the data point's meaning, even if different terminology is used. The goal is semantic accuracy, not literal word matching.\n"
            f"    - **ACCEPTABLE Semantic Matches:** If the source describes the same concept using different but equivalent terms, this is valid. Example: 'embezzlement' can support 'Financial Misconduct'; 'excessive force' can support 'Abuse'.\n"
            f"    - **UNACCEPTABLE Semantic Mismatches:** Reject if the concepts are fundamentally different, even if related. Examples: 'late to work' does not support 'Misconduct'; 'disagreement' does not support 'Abuse'; 'budget cuts' do not support 'Fraud'.\n"
            f"    - **Evaluation Guidelines:** Ask: Do both terms describe essentially the same type of behavior, action, or incident? The source text should provide sufficient context to reasonably categorize the event as described in the data point.\n"
            f"**In your final `notes` field, you must briefly explain how you have confirmed these principles were not violated.**\n\n"

            f"**Core Task: Verify ONE Data Point**\n"
            f"The Source Text below may describe several different events. Your mission is to focus ONLY on the single data point provided. You must find the specific sentence or paragraph that corresponds to this data point and use ONLY that context for verification. Ignore all other events mentioned.\n"

            f"**Primary Purpose Analysis:**\n"
            f"Before checking the facts, you must first determine the primary purpose of the source text. Is it to factually report on a real-world event? Or does it have a different purpose? Data is only considered supported if it comes from a source whose primary purpose is factual reporting.\n\n"
            f"Reject the data point (set `is_accurate` to `False`) if the source text's primary purpose is one of the following:\n"
            f"1.  **Commercial:** The main goal is to sell a product or service.\n"
            f"2.  **Fictional/Creative:** The text is a story, satire, poem, or part of a fictional universe.\n"
            f"3.  **Discussion/Opinion:** The text is user-generated speculation or personal opinion without editorial oversight.\n"
            f"4.  **Metadata/Abstract:** The text is *about* another piece of content, not the event itself.\n\n"
            f"If you reject for one of these reasons, you must state the category in your notes.\n"

            f"**Advanced Date Extraction Rules:**\n"
            f"1.  **Prioritize In-Text Event Dates:** Search for dates mentioned directly with the event's details (e.g.,'...the stock market crash of December 22nd.', 'the fraud occurred between 2014-2016'). Both specific dates and date ranges that describe when an event occurred are acceptable.\n"
            f"2.  **Date Range Handling:** When the source provides a date range for when an event occurred, this is valid temporal information. Examples and formats:\n"
            f"       - Year ranges: 'between 2015-2017' → extract as '2015-2017'\n"
            f"       - Month ranges: 'from March to July 2020' → extract as '2020-03 to 2020-07' or 'March-July 2020'\n"
            f"       - Day ranges: 'between March 15-20, 2020' → extract as '2020-03-15 to 2020-03-20' or 'March 15-20, 2020'\n"
            f"       - Mixed ranges: 'sometime between June 2019 and February 2020' → extract as '2019-06 to 2020-02'\n"
            f"3.  **Reject if Ambiguous:** If you cannot confidently determine the event date from the text (e.g., the only date is a publication date and the event happened 'last week'), you must set `is_accurate` to `False` and explain in the notes that the specific event date is ambiguous.\n"
            f"4.  **Proximity Principle:** The date you extract MUST be mentioned in close proximity (the same sentence or paragraph) to the other facts in the data point. Ignore dates found in unrelated parts of the page or relating to other events.\n"

            f"**CONTENT VALIDITY GUIDELINES:**\n"
            f"- **CRITICAL RULE:** Do NOT mistake a data-rich page (like a directory or list of articles) for a purely navigational one. Your primary goal is to find data, not to judge the page's format.\n"
            f"- **Acceptable Formats:** Meaningful information can be presented as normal paragraph text, data tables, lists, or records.\n"
            f"- **When to Reject:** Only mark `is_article_content` as `False` if the text is purely navigational, contains only error messages, or is completely empty.\n\n"

            f"**CRITICAL RULE FOR INCOMPLETE DATA:**\n"
            f"If a field in the Data Point is empty (null, NaN, or an empty string), you MUST reject the data point. "

            f"**FACT-CHECKING APPROACH:**\n"
            f"1.  Assess if the text has meaningful information using the guidelines and the analysis hint above.\n"
            f"2.  If it does, carefully verify that EVERY value in the Data Point is mentioned and supported by the Source Text, applying the Critical Semantic Audit rules. Remember: look for semantic equivalence, not just exact word matches. The 'source' field does not need verification.\n"
            f"3.  If you find a date for the event in the source text, extract it into the 'extracted_date' field. Use appropriate formats:\n"
            f"       - Specific dates: 'YYYY-MM-DD' (e.g., '2020-03-15')\n"
            f"       - Year ranges: 'YYYY-YYYY' (e.g., '2015-2017')\n"
            f"       - Month ranges: 'YYYY-MM to YYYY-MM' (e.g., '2020-03 to 2020-07') or descriptive (e.g., 'March-July 2020')\n"
            f"       - Day ranges: 'YYYY-MM-DD to YYYY-MM-DD' (e.g., '2020-03-15 to 2020-03-20') or descriptive (e.g., 'March 15-20, 2020')\n\n"
            f"Data Point: {json.dumps(data_point)}\n\n"
            f"Source Text: \"{markdown[:32000]}\""
        )

        try:
            return await self._call_llm_with_structure(prompt, FactCheckerResponse)
        finally:
            self.metrics_tracker.end_agent("fact_checker")


    async def source_scrutinizer_agent(self, data_point: Dict[str, Any]):
        self.metrics_tracker.start_agent("source_scrutinizer")
        url = data_point.get('source')
        if not url or not isinstance(url, str):
            return SourceScrutinizerResponse(source_type="Invalid", reliability="Very Low", notes="Missing or invalid URL.")
        prompt = (
            f"You are a media analyst. Classify the source of the URL '{url}'.\n"
            f"Categorize its type (e.g., 'Major News Outlet', 'Local News', 'Government', 'University', 'Press Release', 'Blog', 'Unknown').\n"
            f"Then, assess its reliability ('Very High', 'High', 'Moderate', 'Low', 'Very Low').\n"
            f"Provide a brief justification."
        )
        try:
            return await self._call_llm_with_structure(prompt, SourceScrutinizerResponse)
        finally:
            self.metrics_tracker.end_agent("source_scrutinizer")

    async def data_formatter_agent(self, data_point: Dict[str, Any]):
        self.metrics_tracker.start_agent("data_formatter")
        prompt = f"Format the following data to conform to the provided Pydantic schema. Ensure all data types and formats are correct.\nData: {json.dumps(data_point)}"
        try:
            return await self._call_llm_with_structure(prompt, self.DataModel)
        finally:
            self.metrics_tracker.end_agent("data_formatter")

    async def relevancy_assessor_agent(self, data_point: dict):
        prompt = f"""
You are a Relevancy Assessor. Your task is to determine if the given data point is TOPICALLY relevant to the dataset description, even if it is incomplete.

**CRITICAL RULE:** Do NOT reject a data point just because it has missing values (like null or NaN). Your only job is to check if the topic matches. Another agent is responsible for filling in missing data. Reject only if the fundamental topic is wrong.

**Dataset Description:** "{self.dataset_description}"

**Data Point to Assess:**
{json.dumps(data_point)}

Based on the **Dataset Description**, is this data point topically relevant, even if fields are missing?
"""
        return await self._call_llm_with_structure(prompt, RelevancyResponse)

    async def arbiter_agent(self, orig_data: dict, fact_report: FactCheckerResponse, source_report: SourceScrutinizerResponse):
        reasons_for_rejection = []
        if fact_report and not fact_report.is_article_content:
            notes = fact_report.notes.lower()
            rejection_type = "INVALID_CONTENT"
            if 'empty_content' in notes: rejection_type = "EMPTY_CONTENT"
            elif 'error_message' in notes: rejection_type = "ERROR_MESSAGE"
            elif 'navigation_only' in notes: rejection_type = "NAVIGATION_ONLY"
            elif 'paywall_blocked' in notes: rejection_type = "PAYWALL_BLOCKED"
            elif 'unrelated_content' in notes: rejection_type = "UNRELATED_CONTENT"
            elif 'parsing_failed' in notes: rejection_type = "PARSING_FAILED"
            reasons_for_rejection.append(f"Fact-Checker: {rejection_type} - {fact_report.notes}")
            # Track error pattern
            self.metrics_tracker.track_error_pattern(rejection_type, "fact_checker")
        if fact_report and fact_report.is_accurate is False:
            reasons_for_rejection.append(f"Fact-Checker: Data not supported by source. Notes: {fact_report.notes}")
            # Track error pattern
            self.metrics_tracker.track_error_pattern("INACCURATE_DATA", "fact_checker")
        if source_report and source_report.reliability in ['Low', 'Very Low']:
            reasons_for_rejection.append(f"Source Scrutinizer: Reliability is '{source_report.reliability}'.")
            # Track error pattern
            self.metrics_tracker.track_error_pattern(f"LOW_RELIABILITY_{source_report.reliability.upper()}", "source_scrutinizer")
        if reasons_for_rejection:
            decision = "REJECT"
            reasoning = " | ".join(reasons_for_rejection)
        else:
            decision = "ACCEPT"
            reasoning = "All checks passed."
        return ArbiterResponse(decision=decision, reasoning=reasoning)

    async def _run_fact_lookup_tool(self, question: str) -> str:
        self.logger.info(f"Running Resilient Fact Lookup Tool for question: '{question}'")
        try:
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: scrape_google_search_results(question, num_pages=1, num_results=5)
            )

            if not search_results:
                self.logger.warning(f"Fact lookup failed for '{question}': No search results found.")
                return "Fact lookup failed: No search results found."

            for i, url in enumerate(search_results):
                self.logger.info(f"Attempting extraction from source {i+1}: {url}")
                try:
                    content = await self.content_manager.get_content(url)
                    if not content or len(content) < 50:
                        self.logger.warning(f"Skipping URL, insufficient content: {url}")
                        continue
                
                    extraction_prompt = f"""
                    From the following text, extract the direct and precise answer to the question: "{question}".
                    Prioritize answers that appear to come from authoritative sources (e.g., census data, official reports, data tables).
                    Provide only the answer itself, with no extra words.
                    If the answer is not clearly present in the text, respond with only the phrase "Answer not found."

                    Text:
                    "{content[:8000]}"
                    """

                    answer = await self._call_llm(extraction_prompt)

                    if "Answer not found" not in answer:
                        self.logger.info(f"Successfully extracted answer '{answer}' from {url}")
                        return answer
                except Exception as e:
                    self.logger.error(f"Error processing URL {url}", extra={'error': str(e)})
                    continue

            self.logger.warning(f"Fact lookup failed for '{question}': Answer not found in top {len(search_results)} sources.")
            return "Answer not found in top search results."

        except Exception as e:
            self.logger.error("FactLookupTool failed unexpectedly", extra={'question': question, 'error': str(e)})
            return f"Fact lookup failed with an error: {e}"

    async def data_remediation_agent_same(self, data_point: Dict[str, Any], rejection_reason: str, page_markdown: str):
        context_rules = self.meta_context_agent.generate_context_rules()

        analyst_prompt = f"""
You are a methodical data analyst. Your only job is to analyze the provided context and create a plan to fix the rejected data point. Do not perform the fix yourself.

{context_rules}

**CONTEXT**
- **Rejected Data Point:** {json.dumps(data_point)}
- **Rejection Reason:** "{rejection_reason}"
- **Source Text:** "{page_markdown[:10000]}"

**YOUR TASK**
1.  Read the rejection reason to see what's wrong.
2.  Scan the source text to see how it can be fixed.
3.  Determine if the fix is a **direct value** found in the text or if it requires a **calculation** using a relative value (like a percentage or a date).
4.  Fill out the fields in the `RemediationAnalystResponse` to describe your plan. If a calculation is needed, formulate the precise question for the tool.

**CRITICAL: PRESERVE DATA TYPES**
- If the original field was numeric, keep the remediated value numeric
- For uncertain numbers, use the base numeric value (e.g., 4000000 not "approximately 4 million")
- Document uncertainty in reasoning, not in the data value itself

**SPECIAL CASE: Missing vs Zero Values**
- Treat NaN/null downloads fields the same as 0 when text contains percentages
- Both cases can be resolved through population-based calculations
"""
        analysis_report = await self._call_llm_with_structure(analyst_prompt, RemediationAnalystResponse)

        if not analysis_report:
            return RemediationSameResponse(status="FAILED", updated_data=None, reasoning="The Analyst Agent failed to produce a valid plan.")

        if analysis_report.is_direct_fix_possible and analysis_report.direct_answer:
            updated_field_key = next(iter(data_point.keys() - {'source', 'id', 'date'}))
            return RemediationSameResponse(
                status="SUCCESS",
                updated_data={updated_field_key: analysis_report.direct_answer},
                reasoning=analysis_report.analysis
            )

        elif analysis_report.is_calculation_needed and analysis_report.question_for_tool:
            self.logger.info(f"Analyst determined a tool is needed. Question: {analysis_report.question_for_tool}")
            tool_result = await self._run_fact_lookup_tool(analysis_report.question_for_tool)

            if "not found" in tool_result.lower():
                return RemediationSameResponse(status="FAILED", updated_data=None, reasoning=f"Fact-lookup tool failed for question: '{analysis_report.question_for_tool}'.")
            
            calculator_prompt = f"""
You are a calculator bot. Your only job is to perform the math described in the 'Analysis' and return the final, corrected data point.

**Analysis from previous step:**
{analysis_report.analysis}

**Information from Fact-Lookup Tool:**
The answer to "{analysis_report.question_for_tool}" is: {tool_result}

**Original Data Point:**
{json.dumps(data_point)}

Perform the calculation and provide the result in the `RemediationSameResponse` format.
"""
            return await self._call_llm_with_structure(calculator_prompt, RemediationSameResponse)

        else:
            return RemediationSameResponse(status="FAILED", updated_data=None, reasoning=analysis_report.analysis)

    async def data_remediation_agent_different(self, page_markdown: str):
        self.metrics_tracker.start_agent("remediation_different")
        self.metrics_tracker.record_discovery_attempt()
        _start = datetime.now()
        if not self.DataModel:
            return RemediationDifferentResponse(status="FAILED", found_data=[], reasoning="DataModel not initialized.")

        context_rules = self.meta_context_agent.generate_context_rules()

        identification_prompt = f"""
You are a text analysis expert. Your task is to read the following source text and identify every single sentence or short, self-contained line-item that appears to contain a complete data point matching the described schema.

{context_rules}

**Data Schema to look for:**
{json.dumps(self.DataModel.schema(), indent=2)}

**Source Text:**
"{page_markdown[:32000]}"

**Instructions:**
- Return ONLY a JSON object with a single key "potential_sentences", which holds a Python list of the raw sentence strings you found.
- If you find nothing, return an empty list: {{"potential_sentences": []}}
- Do not attempt to format the data points yet. Just extract the sentences.

Example Output:
{{
  "potential_sentences": [
    "A stage 4 hurricane struck the US on Saturday, Aug. 12, 2003, killing more than 20 people.",
    "On September 1, a thunderstorm destroyed upwards of 350 homes and killed 3 in the Ouest Department."
  ]
}}
"""

        response_str = await self._call_llm(identification_prompt)
        
        try:
            potential_sentences = json.loads(response_str).get("potential_sentences", [])
            if not potential_sentences:
                return RemediationDifferentResponse(status="FAILED", found_data=[], reasoning="Could not identify any potential data points in the source text.")
        except (json.JSONDecodeError, AttributeError):
            if self.logger:
                self.logger.error("Failed to parse sentence identification response.", extra={'raw_content': response_str})
            return RemediationDifferentResponse(status="FAILED", found_data=[], reasoning="Failed to parse sentence list from the identification step.")

        discovered_data_points = []
        for sentence in potential_sentences:
            extraction_prompt = f"""
You are a precise data extraction bot. From the single sentence provided below, extract one data point that strictly conforms to the given JSON schema.

{context_rules}

**JSON Schema:**
{json.dumps(self.DataModel.schema(), indent=2)}

**Sentence to Extract From:**
"{sentence}"

**Instructions:**
- If you can extract a valid data point, return it as a single JSON object.
- If the sentence does not contain all the necessary information to fill the schema, return the word "null" and nothing else.
- Follow the dataset-specific context rules above when extracting entities and dates.
"""

            try:
                extraction_response_str = await self._call_llm(extraction_prompt)
                if "null" in extraction_response_str.lower():
                    continue

                if '```json' in extraction_response_str:
                    extraction_response_str = extraction_response_str.split('```json\n', 1)[1].split('\n```', 1)[0]

                data_point = json.loads(extraction_response_str)
                self.DataModel.parse_obj(data_point)
                discovered_data_points.append(data_point)
            except (json.JSONDecodeError, tenacity.RetryError, Exception):
                continue

        if discovered_data_points:
            duration_s = (datetime.now() - _start).total_seconds()
            self.metrics_tracker.record_discovery_result(len(discovered_data_points), duration_s, 0.0)
            return RemediationDifferentResponse(
                status="SUCCESS",
                found_data=discovered_data_points,
                reasoning=f"Successfully discovered and extracted {len(discovered_data_points)} new data point(s) from the page."
            )
        else:
            return RemediationDifferentResponse(
                status="FAILED",
                found_data=[],
                reasoning="Identified potential sentences, but failed to extract any valid, complete data points from them."
            )
        
        self.metrics_tracker.end_agent("remediation_different")

    async def remediation_auditor_agent(self, data_point: dict, page_markdown: str, remediation_reasoning: str):
        """
        Audits a previously remediated data point by checking if the reasoning for the
        correction is sound and consistent with the source text.
        """
        prompt = f"""
You are a meticulous auditor. A data point has been programmatically corrected using an external tool, and your job is to validate this correction.

**CRITICAL CONTEXT**
- **Remediated Data Point:** {json.dumps(data_point)}
- **Original Source Text:** "{page_markdown[:8000]}"
- **Reasoning for the Correction:** "{remediation_reasoning}"

**YOUR TASK: AUDIT THE REASONING**
Your goal is NOT to find the new value in the Original Source Text. We know it's not there. Your job is to answer one question:

**Is the 'Reasoning for the Correction' a logical and consistent explanation for how the 'Remediated Data Point' was derived from the 'Original Source Text'?**

1.  Check if the premise for the calculation exists in the source text (e.g., does the text actually mention the percentage or relative value that the reasoning claims to use?).
2.  Check if the calculation described in the reasoning makes mathematical sense.
3.  Check if the final value in the data point is the result of that sound reasoning.

If all three checks pass, you must set `is_accurate` to `true`. Your notes should summarize why you are accepting the externally sourced value (e.g., "Accepting calculated value based on valid reasoning and premises found in the source text.").
"""
        return await self._call_llm_with_structure(prompt, FactCheckerResponse)

    async def data_integrity_validator_agent(self, data_point: dict):
        """
        Checks a data point for completeness and semantic plausibility based on its schema.
        """
        prompt = f"""
You are a simple, literal data checker bot. Your job is to perform two final, simple checks on the data point.

**CRITICAL COMMAND: You have ONLY TWO rules. You must not perform any other kind of analysis. Do not check dates. Another system has already validated all of those things. Your only job is to check for the two rules below.**

---
**CONTEXT**

1.  **Required Schema Fields:** {json.dumps(list(self.DataModel.__fields__.keys()))}
2.  **Field Descriptions:**
    {json.dumps(self.DataModel.schema()['properties'], indent=2)}

---
**DATA POINT TO VALIDATE**

{json.dumps(data_point, indent=2)}

---
**YOUR ONLY TWO RULES**

1.  **Is any REQUIRED field empty?**
    * Look at the list of "Required Schema Fields" above.
    * Check if any of those specific fields in the data point are missing, null, NaN, or an empty string ("").
    * (You can ignore fields that are not in the required list)

2.  **Is any field's text obvious nonsense?**
    * Read the field's description. Does the value make sense?
    * This is not a deep check. You are only looking for clear, obvious mistakes.
    * Example: If a field description is "The name of a virus," a value of the number `2` or the text "asdfghjkl" is obvious nonsense. A value of "Flu" is NOT nonsense.

Based ONLY on a direct violation of one of these two rules, is the data point plausible?
"""
        return await self._call_llm_with_structure(prompt, DataIntegrityResponse)

async def process_row(committee, row, all_columns):
    """
    Processes a single data point through the AI Committee pipeline.
    """
    orig = row if isinstance(row, dict) else row.to_dict()
    row_id = f"Row_{hash(json.dumps(orig, sort_keys=True)) % 10000:04d}"
    log_entry = {
        "row_id": row_id,
        "original_data": orig,
        "processing_start": datetime.now().isoformat()
    }

    try:
        committee.metrics_tracker.start_row_processing(row_id)
    except Exception:
        pass

    final_decision_str = ""
    final_reasoning_str = ""
    final_data = {}

    current_data = orig.copy()
    page_markdown = None
    remediation_context = None

    for i in range(2):
        log_entry[f"pass_{i+1}_data"] = current_data

        committee.set_metrics_context(row_id, "relevancy_assessor")
        relevancy_report = await committee.relevancy_assessor_agent(current_data)
        log_entry[f"pass_{i+1}_relevancy_report"] = relevancy_report.dict() if relevancy_report else None
        if not relevancy_report or not relevancy_report.is_relevant:
            final_decision_str = "REJECT"
            final_reasoning_str = f"Relevancy Check Failed: {relevancy_report.reasoning if relevancy_report else 'Agent failed.'}"
            break

        if page_markdown is None:
            url = current_data.get('source')
            if not url or not isinstance(url, str) or not url.startswith('http'):
                final_decision_str = "REJECT"
                final_reasoning_str = "Invalid or missing source URL."
                committee.metrics_tracker.track_error_pattern("INVALID_URL", "url_validation")
                break
            committee.set_metrics_context(row_id, "crawler")
            page_markdown = await committee.content_manager.get_content(url)
            if page_markdown is None:
                final_decision_str = "REJECT"
                final_reasoning_str = f"Crawl failed for source URL: {url}. Cannot perform validation."
                committee.metrics_tracker.track_error_pattern("CRAWL_FAILED", "content_manager")
                break

        committee.set_metrics_context(row_id, "layout_analyzer")
        layout_report = await committee.layout_analyzer_agent(page_markdown)

        if i == 1 and remediation_context:
            committee.set_metrics_context(row_id, "remediation_auditor")
            fact_report = await committee.remediation_auditor_agent(current_data, page_markdown, remediation_context)
        else:
            committee.set_metrics_context(row_id, "fact_checker")
            fact_report = await committee.fact_checker_agent_from_content(current_data, page_markdown, layout_report)

        if fact_report and fact_report.extracted_date:
            try:
                extracted_dt = date_parse(fact_report.extracted_date).date()
                if extracted_dt > datetime.now().date():
                    fact_report.is_accurate = False
                    fact_report.notes += " | TEMPORAL_CHECK_FAILED: The extracted date is in the future."
            except (ValueError, TypeError):
                pass
            
        committee.set_metrics_context(row_id, "source_scrutinizer")
        source_report = await committee.source_scrutinizer_agent(current_data)

        log_entry[f"pass_{i+1}_reports"] = {
            "layout_analysis_report": layout_report.dict() if layout_report else None,
            "source_scrutinizer_report": source_report.dict() if source_report else None,
            "fact_checker_report": fact_report.dict() if fact_report else None
        }

        arbiter_decision = await committee.arbiter_agent(current_data, fact_report, source_report)
        log_entry[f"pass_{i+1}_arbiter_decision"] = arbiter_decision.dict() if arbiter_decision else None

        if arbiter_decision and arbiter_decision.decision == "ACCEPT":
            final_decision_str = "ACCEPT" if i == 0 else "ACCEPT (After Remediation)"
            final_reasoning_str = arbiter_decision.reasoning
            committee.set_metrics_context(row_id, "data_formatter")
            formatter_output = await committee.data_formatter_agent(current_data)
            final_data = current_data.copy()
            if formatter_output:
                final_data.update(formatter_output.dict())
            break

        if i == 0:
            rejection_reason = arbiter_decision.reasoning if arbiter_decision else "Unknown reason"

            committee.set_metrics_context(row_id, "remediation_same")
            remediation_report_same = await committee.data_remediation_agent_same(current_data, rejection_reason, page_markdown)
            log_entry["remediation_report_same"] = remediation_report_same.dict() if remediation_report_same else None

            if remediation_report_same and remediation_report_same.status == 'SUCCESS':
                remediated_data = current_data.copy()
                if remediation_report_same.updated_data:
                    remediated_data.update(remediation_report_same.updated_data)

                try:
                    committee.DataModel.parse_obj(remediated_data)
                    current_data = remediated_data
                    remediation_context = remediation_report_same.reasoning
                    # Track successful remediation
                    committee.metrics_tracker.track_remediation(success=True, remediation_type="same")
                    continue
                except Exception as e:
                    final_decision_str = "REJECT"
                    final_reasoning_str = f"Remediation (Same) failed schema validation: {e}"
                    # Track failed remediation
                    committee.metrics_tracker.track_remediation(success=False, remediation_type="same")
                    break

            committee.set_metrics_context(row_id, "remediation_different")
            remediation_report_different = await committee.data_remediation_agent_different(page_markdown)
            log_entry["remediation_report_different"] = remediation_report_different.dict()
            if remediation_report_different and remediation_report_different.status == 'SUCCESS' and remediation_report_different.found_data:
                # Track successful remediation (different)
                committee.metrics_tracker.track_remediation(success=True, remediation_type="different")
                newly_found_data = remediation_report_different.found_data
                original_source = current_data.get('source')
                for point in newly_found_data:
                    point['source'] = original_source

                log_entry["final_decision"] = "DISCOVERED_NEW"
                log_entry["final_reasoning"] = f"Found {len(newly_found_data)} new data points on the page."
                log_entry["processing_end"] = datetime.now().isoformat()
                committee.logger.info("Discovered new data points", extra={'details': log_entry})
                try:
                    committee.metrics_tracker.end_row_processing(row_id, "DISCOVERED_NEW", log_entry["final_reasoning"])
                except Exception:
                    pass
                return {"status": "DISCOVERED_NEW", "data": newly_found_data}

            final_decision_str = "REJECT"
            final_reasoning_str = f"All remediation attempts failed. Original reason: {rejection_reason}"
            break
        else:
            final_decision_str = "REJECT"
            final_reasoning_str = f"Remediated data was also rejected. Reason: {arbiter_decision.reasoning if arbiter_decision else 'Unknown'}"
            break

    log_entry["final_decision"] = final_decision_str
    log_entry["final_reasoning"] = final_reasoning_str
    log_entry["processing_end"] = datetime.now().isoformat()
    committee.logger.info(f"Row processed with final decision: {final_decision_str}", extra={'details': log_entry})

    if final_decision_str.startswith("ACCEPT"):
        aligned_data = {}
        for col in all_columns:
            aligned_data[col] = final_data.get(col, orig.get(col, None))
        try:
            committee.metrics_tracker.end_row_processing(row_id, final_decision_str, final_reasoning_str)
        except Exception:
            pass
        return {"status": "ACCEPT", "data": aligned_data}
    else:
        try:
            committee.metrics_tracker.end_row_processing(row_id, "REJECT", final_reasoning_str)
        except Exception:
            pass
        return {"status": "REJECT", "reason": final_reasoning_str}

async def process_row_with_semaphore(committee, row, all_columns, semaphore):
    async with semaphore:
        try:
            return await process_row(committee, row, all_columns)
        except (tenacity.RetryError, Exception) as e:
            error_type = "Persistent API Error" if isinstance(e, tenacity.RetryError) else "Unexpected Processing Error"
            log_details = {
                "row_id": f"Row_{hash(json.dumps(row, sort_keys=True)) % 10000:04d}",
                "original_data": row,
                "final_decision": f"SKIP ({error_type})",
                "final_reasoning": str(e)
            }
            try:
                committee.metrics_tracker.log_failure(log_details["row_id"], "pipeline")
                committee.metrics_tracker.end_row_processing(log_details["row_id"], log_details["final_decision"], log_details["final_reasoning"])
            except Exception:
                pass
            committee.logger.error(f"Row skipped due to {error_type}", extra={'details': log_details})
            return {"status": "SKIP", "reason": str(e)}

async def main(args):
    log_file_path = os.path.splitext(args.output_file)[0] + "_comprehensive.log"
    logger = setup_logger(log_file_path)
    if not os.path.exists(args.input_file):
        print(f"error: input file not found: {args.input_file}")
        return
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_file)
    all_df_columns = df.columns.tolist()
    
    metrics = ComprehensiveMetrics(price_input=args.price_input, price_output=args.price_output, price_cached=getattr(args, 'price_cached', args.price_input))
    semaphore = asyncio.Semaphore(args.concurrency)
    committee = AICommittee(llm_model_name=args.model, metrics_tracker=metrics, logger=logger)
    await committee._initialize_framework(args.dataset_description, args.schema_fields, df)

    candidates = df.to_dict('records')
    accepted_results = []
    processed_hashes = set()
    iteration_count = 1

    while candidates:

        unique_candidates = []
        for cand in candidates:
            cand_hash = hashlib.sha256(json.dumps(cand, sort_keys=True).encode()).hexdigest()
            if cand_hash not in processed_hashes:
                unique_candidates.append(cand)
                processed_hashes.add(cand_hash)

        if not unique_candidates:
            break

        tasks = [process_row_with_semaphore(committee, row, all_df_columns, semaphore) for row in unique_candidates]
        results = await tqdm_asyncio.gather(*tasks)

        next_round_candidates = []
        for res in results:
            if not res:
                continue
            if res['status'] == 'ACCEPT':
                accepted_results.append(res['data'])
            elif res['status'] == 'DISCOVERED_NEW':
                next_round_candidates.extend(res['data'])
        
        candidates = next_round_candidates
        iteration_count += 1



    deduplicated_results = hierarchical_deduplication(accepted_results, args.schema_fields)

    for dp in deduplicated_results:
        dp.pop('potential_issues', None)

    async def validate_integrity(data_point):
        async with semaphore:
            integrity_report = await committee.data_integrity_validator_agent(data_point)
            if integrity_report and integrity_report.is_plausible:
                return data_point
            else:
                reason = integrity_report.reasoning if integrity_report else "Agent failed"
                log_details = {
                    "row_id": f"Final_Validation_{hash(str(data_point)) % 10000:04d}",
                    "original_data": data_point,
                    "final_decision": "REJECT (Final Integrity Check)",
                    "final_reasoning": f"Data point failed final integrity check: {reason}"
                }
                logger.info("Data point rejected during final integrity validation", extra={'details': log_details})
                return None

    integrity_tasks = [validate_integrity(dp) for dp in deduplicated_results]
    final_validated_results = await tqdm_asyncio.gather(*integrity_tasks)

    final_results_to_save = [res for res in final_validated_results if res is not None]

    await committee.content_manager.close()
    
    if final_results_to_save:
        final_df = pd.DataFrame(final_results_to_save, columns=all_df_columns)
        try:
            with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
                final_df.to_csv(f, index=False)
            print(f"saved {len(final_results_to_save)} rows to {args.output_file}")
        except IOError as e:
            print(f"error writing to file: {args.output_file}: {e}")
    else:
        print("no data points passed final validation")

    # Persist metrics in both human-readable and machine-readable forms
    metrics.report()
    extras = {
        "architecture": "multi_agent",
        "dataset": args.input_file,
        "model": args.model,
        "concurrency": args.concurrency,
        "processing_stats": {
            "initial_rows": len(df),
            "final_validated_rows": len(final_results_to_save),
        },
    }
    metrics_json_path = os.path.splitext(args.output_file)[0] + "_metrics.json"
    metrics_csv_path = os.path.splitext(args.output_file)[0] + "_metrics.csv"
    metrics.save_json(metrics_json_path, extras=extras)
    metrics.save_csv(metrics_csv_path, extras=extras)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Automated Data Evaluator using an AI Committee.")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_file", required=True, help="Path for the output validated CSV file.")
    parser.add_argument("-d", "--dataset_description", required=True, help="A clear, one-sentence description of the data.")
    parser.add_argument("-f", "--schema_fields", required=True, nargs='+', help="A list of field names in the dataset (e.g., name city state).")
    parser.add_argument("-m", "--model", default="gpt-4o-mini", help="Name of the LLM to use.")
    parser.add_argument("-c", "--concurrency", type=int, default=3, help="Max concurrent tasks.")
    parser.add_argument("--price_input", type=float, default= 0.15 / 1_000_000, help="Price per input token (USD). Example: $1.10 per million.")
    parser.add_argument("--price_cached", type=float, default= 0.075 / 1_000_000, help="Price per cached token (USD). Example: $0.075 per million.")
    parser.add_argument("--price_output", type=float, default= 0.60 / 1_000_000, help="Price per output token (USD). Example: $4.40 per million.")
    args = parser.parse_args()
    asyncio.run(main(args))