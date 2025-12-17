"""
Prompt Templates Module

Manages and renders prompt templates with variable substitution.
Integrates with LangChain PromptTemplates.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages prompt templates with caching and validation"""
    
    def __init__(self):
        self._templates: Dict[str, Dict[str, str]] = {}
        self._langchain_templates: Dict[str, ChatPromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from YAML config"""
        try:
            from config import settings
            self._templates = settings._prompt_templates
            logger.info(f"Loaded prompt templates from config")
        except Exception as e:
            logger.warning(f"Could not load templates from config: {e}")
            self._templates = {}
    
    def get_template(self, category: str, name: str) -> str:
        """Get a raw template string"""
        try:
            return self._templates[category][name]
        except KeyError:
            raise KeyError(f"Template not found: {category}.{name}")
    
    def render(
        self,
        category: str,
        name: str,
        **variables
    ) -> str:
        """Render a template with variables"""
        template = self.get_template(category, name)
        
        # Use format-style substitution
        try:
            return template.format(**variables)
        except KeyError as e:
            # Try partial substitution
            for key, value in variables.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
    
    def get_langchain_template(
        self,
        category: str,
        name: str
    ) -> ChatPromptTemplate:
        """Get or create a LangChain ChatPromptTemplate"""
        cache_key = f"{category}.{name}"
        
        if cache_key not in self._langchain_templates:
            template_str = self.get_template(category, name)
            
            # Convert to LangChain format
            # Replace {var} with {var} (LangChain uses same format)
            self._langchain_templates[cache_key] = PromptTemplate.from_template(
                template_str
            )
        
        return self._langchain_templates[cache_key]
    
    def create_chat_template(
        self,
        system_template: str,
        human_template: str,
        input_variables: List[str] = None
    ) -> ChatPromptTemplate:
        """Create a ChatPromptTemplate from system and human templates"""
        
        # Auto-detect input variables if not provided
        if input_variables is None:
            input_variables = self._extract_variables(system_template)
            input_variables.extend(self._extract_variables(human_template))
            input_variables = list(set(input_variables))
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template"""
        pattern = r'\{(\w+)\}'
        return list(set(re.findall(pattern, template)))
    
    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates by category"""
        result = {}
        for category, templates in self._templates.items():
            if isinstance(templates, dict):
                result[category] = list(templates.keys())
        return result


class VibePromptBuilder:
    """Specialized prompt builder for vibe generation"""
    
    def __init__(self, template_manager: PromptTemplateManager = None):
        self.tm = template_manager or PromptTemplateManager()
    
    def build_freeform_prompt(
        self,
        product_name: str,
        product_type: str,
        description: str,
        colors: str = "N/A",
        material: str = "N/A",
        price: float = 0,
        currency: str = "MYR"
    ) -> ChatPromptTemplate:
        """Build prompt for free-form vibe generation"""
        
        system = self.tm.get_template("vibe_generator", "freeform")["system"]
        user = self.tm.get_template("vibe_generator", "freeform")["user"]
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user)
        ]).partial(
            product_name=product_name,
            product_type=product_type,
            description=description,
            colors=colors,
            material=material,
            price=price,
            currency=currency
        )
    
    def build_batch_prompt(
        self,
        products: List[Dict[str, Any]]
    ) -> ChatPromptTemplate:
        """Build prompt for batch vibe generation"""
        import json
        
        system = self.tm.get_template("vibe_generator", "freeform")["system"]
        user = self.tm.get_template("vibe_generator", "freeform")["batch_user"]
        
        products_json = json.dumps([
            {
                "product_id": p.get("product_id", ""),
                "product_name": p.get("product_name", ""),
                "product_type": p.get("product_type", ""),
                "colors": p.get("colors_available", ""),
                "material": p.get("material", "N/A"),
                "description": p.get("product_description", "")[:400]
            }
            for p in products
        ], indent=2)
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user)
        ]).partial(
            count=len(products),
            products_json=products_json
        )
    
    def build_hybrid_prompt(
        self,
        product_name: str,
        description: str,
        colors: str,
        material: str,
        rule_based_tags: List[str]
    ) -> ChatPromptTemplate:
        """Build prompt for hybrid vibe enhancement"""
        
        system = self.tm.get_template("vibe_generator", "hybrid_enhance")["system"]
        user = self.tm.get_template("vibe_generator", "hybrid_enhance")["user"]
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user)
        ]).partial(
            product_name=product_name,
            description=description,
            colors=colors,
            material=material,
            rule_based_tags=", ".join(rule_based_tags)
        )


# Singleton instances
_template_manager: Optional[PromptTemplateManager] = None
_vibe_prompt_builder: Optional[VibePromptBuilder] = None


def get_template_manager() -> PromptTemplateManager:
    """Get singleton template manager"""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager


def get_vibe_prompt_builder() -> VibePromptBuilder:
    """Get singleton vibe prompt builder"""
    global _vibe_prompt_builder
    if _vibe_prompt_builder is None:
        _vibe_prompt_builder = VibePromptBuilder(get_template_manager())
    return _vibe_prompt_builder
