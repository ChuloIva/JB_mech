"""
Attack template extraction using Trinity LLM via OpenRouter.

Extracts structured attack patterns from vulnerability descriptions.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import openai
from tqdm import tqdm


@dataclass
class AttackTemplate:
    """Structured attack template extracted from vulnerability."""
    vuln_id: str
    vuln_title: str
    attack_type: str
    setup_steps: List[str]
    example_payloads: List[str]
    bypass_techniques: List[str]
    preconditions: List[str]
    target_models: List[str]
    success_indicators: List[str]
    raw_response: str


@dataclass
class TemplateExtractionConfig:
    """Configuration for template extraction."""
    model: str = "arcee-ai/trinity-large-preview:free"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 1500
    temperature: float = 0.3
    rate_limit_delay: float = 1.0  # seconds between requests
    batch_save_interval: int = 10
    cache_dir: Path = Path("outputs/vuln_analysis/templates_cache")


TEMPLATE_EXTRACTION_PROMPT = '''Analyze this LLM vulnerability and extract a structured attack template.

VULNERABILITY:
Title: {title}
Tags: {tags}
Affected Models: {models}
Description: {description}

Extract and return a JSON object with these fields:
{{
    "attack_type": "primary attack category (jailbreak/injection/extraction/poisoning/etc)",
    "setup_steps": ["step 1 to prepare the attack", "step 2", ...],
    "example_payloads": ["example prompt/payload 1", "example 2", ...],
    "bypass_techniques": ["technique used to bypass safety 1", ...],
    "preconditions": ["required condition 1", "required condition 2", ...],
    "success_indicators": ["how to know if attack succeeded 1", ...]
}}

Return ONLY valid JSON, no explanation.'''


class AttackTemplateExtractor:
    """
    Extract structured attack templates using Trinity LLM.

    Handles rate limiting and caching for free-tier API.
    """

    def __init__(self, config: Optional[TemplateExtractionConfig] = None):
        self.config = config or TemplateExtractionConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
        )

    def _get_cache_path(self, vuln_id: str) -> Path:
        """Get cache path for a vulnerability."""
        # Sanitize ID for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in vuln_id)
        return self.config.cache_dir / f"{safe_id}.json"

    def _extract_single(
        self,
        vuln: Dict,
    ) -> Optional[AttackTemplate]:
        """Extract template from single vulnerability."""
        vuln_id = vuln.get("cveId", vuln.get("slug", "unknown"))

        # Check cache
        cache_path = self._get_cache_path(vuln_id)
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            return AttackTemplate(**cached)

        prompt = TEMPLATE_EXTRACTION_PROMPT.format(
            title=vuln.get("title", ""),
            tags=", ".join(vuln.get("tags", [])),
            models=", ".join(vuln.get("affectedModels", [])[:10]),
            description=vuln.get("description", "")[:2000],  # Truncate long descriptions
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            raw_response = response.choices[0].message.content.strip()

            # Parse JSON from response
            try:
                json_start = raw_response.find("{")
                json_end = raw_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    extracted = json.loads(raw_response[json_start:json_end])
                else:
                    extracted = {}
            except json.JSONDecodeError:
                extracted = {}

            template = AttackTemplate(
                vuln_id=vuln_id,
                vuln_title=vuln.get("title", ""),
                attack_type=extracted.get("attack_type", "unknown"),
                setup_steps=extracted.get("setup_steps", []),
                example_payloads=extracted.get("example_payloads", []),
                bypass_techniques=extracted.get("bypass_techniques", []),
                preconditions=extracted.get("preconditions", []),
                target_models=vuln.get("affectedModels", []),
                success_indicators=extracted.get("success_indicators", []),
                raw_response=raw_response,
            )

            # Cache result
            with open(cache_path, "w") as f:
                json.dump(asdict(template), f, indent=2)

            return template

        except Exception as e:
            print(f"Error extracting template for {vuln_id}: {e}")
            return None

    def extract_all(
        self,
        vulnerabilities: List[Dict],
        show_progress: bool = True,
        max_items: Optional[int] = None,
    ) -> List[AttackTemplate]:
        """
        Extract templates from all vulnerabilities with rate limiting.

        Args:
            vulnerabilities: List of vulnerability dictionaries
            show_progress: Show progress bar
            max_items: Limit number of items to process (for testing)

        Returns:
            List of extracted templates
        """
        templates = []

        items = vulnerabilities[:max_items] if max_items else vulnerabilities

        iterator = enumerate(items)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Extracting templates")

        for i, vuln in iterator:
            template = self._extract_single(vuln)
            if template:
                templates.append(template)

            # Rate limiting for free tier
            if i > 0 and i % self.config.batch_save_interval == 0:
                self._save_batch(templates)

            # Don't sleep if we hit cache
            cache_path = self._get_cache_path(
                vuln.get("cveId", vuln.get("slug", "unknown"))
            )
            if not cache_path.exists():
                time.sleep(self.config.rate_limit_delay)

        self._save_batch(templates)
        return templates

    def _save_batch(self, templates: List[AttackTemplate]):
        """Save current batch of templates."""
        output_path = self.config.cache_dir.parent / "attack_templates.json"
        with open(output_path, "w") as f:
            json.dump([asdict(t) for t in templates], f, indent=2)

    def get_cached_count(self) -> int:
        """Return number of cached templates."""
        return len(list(self.config.cache_dir.glob("*.json")))
