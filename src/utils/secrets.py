"""
Cloud Secrets Manager

Unified interface for retrieving API keys from various cloud providers.
Supports: AWS, GCP, Azure, HashiCorp Vault, and local .env fallback.

Usage:
    from src.utils.secrets import load_secrets
    
    # Auto-detect cloud provider and load secrets
    load_secrets()
    
    # Or specify provider
    load_secrets(provider="aws", secret_name="bynoemie/api-keys")
"""

import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """Abstract base class for secrets providers"""
    
    @abstractmethod
    def get_secrets(self, secret_name: str) -> Dict[str, str]:
        """Retrieve secrets as a dictionary"""
        pass


class AWSSecretsManager(SecretsProvider):
    """AWS Secrets Manager provider"""
    
    def __init__(self, region: str = None):
        self.region = region or os.getenv("AWS_REGION", "ap-southeast-1")
    
    def get_secrets(self, secret_name: str) -> Dict[str, str]:
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            client = boto3.client('secretsmanager', region_name=self.region)
            response = client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                # Binary secret
                import base64
                return json.loads(base64.b64decode(response['SecretBinary']))
                
        except ClientError as e:
            logger.error(f"AWS Secrets Manager error: {e}")
            raise
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")


class GCPSecretManager(SecretsProvider):
    """Google Cloud Secret Manager provider"""
    
    def __init__(self, project_id: str = None):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
    
    def get_secrets(self, secret_name: str) -> Dict[str, str]:
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            
            # If secret_name contains multiple keys, it's a JSON secret
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            # Try to parse as JSON, otherwise return as single key
            try:
                return json.loads(secret_value)
            except json.JSONDecodeError:
                return {secret_name: secret_value}
                
        except Exception as e:
            logger.error(f"GCP Secret Manager error: {e}")
            raise


class AzureKeyVault(SecretsProvider):
    """Azure Key Vault provider"""
    
    def __init__(self, vault_url: str = None):
        self.vault_url = vault_url or os.getenv("AZURE_VAULT_URL")
    
    def get_secrets(self, secret_name: str) -> Dict[str, str]:
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=self.vault_url, credential=credential)
            
            # Get secret
            secret = client.get_secret(secret_name)
            
            # Try to parse as JSON
            try:
                return json.loads(secret.value)
            except json.JSONDecodeError:
                return {secret_name: secret.value}
                
        except Exception as e:
            logger.error(f"Azure Key Vault error: {e}")
            raise


class HashiCorpVault(SecretsProvider):
    """HashiCorp Vault provider"""
    
    def __init__(self, url: str = None, token: str = None):
        self.url = url or os.getenv("VAULT_ADDR")
        self.token = token or os.getenv("VAULT_TOKEN")
    
    def get_secrets(self, secret_name: str) -> Dict[str, str]:
        try:
            import hvac
            
            client = hvac.Client(url=self.url, token=self.token)
            
            if not client.is_authenticated():
                raise ValueError("Vault authentication failed")
            
            # Read from KV v2
            secret = client.secrets.kv.v2.read_secret_version(path=secret_name)
            return secret['data']['data']
            
        except ImportError:
            raise ImportError("Install hvac: pip install hvac")
        except Exception as e:
            logger.error(f"HashiCorp Vault error: {e}")
            raise


class LocalEnvProvider(SecretsProvider):
    """Local .env file provider (fallback)"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
    
    def get_secrets(self, secret_name: str = None) -> Dict[str, str]:
        secrets = {}
        
        if self.env_file.exists():
            try:
                from dotenv import dotenv_values
                secrets = dotenv_values(self.env_file)
            except ImportError:
                # Manual parsing
                with open(self.env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, _, value = line.partition('=')
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            secrets[key] = value
        
        return dict(secrets)


class SecretsManager:
    """
    Unified secrets manager with auto-detection and fallback.
    
    Usage:
        manager = SecretsManager()
        manager.load_to_env()  # Loads all secrets to environment
        
        # Or get specific secret
        api_key = manager.get("GROQ_API_KEY")
    """
    
    PROVIDER_MAP = {
        "aws": AWSSecretsManager,
        "gcp": GCPSecretManager,
        "azure": AzureKeyVault,
        "vault": HashiCorpVault,
        "local": LocalEnvProvider,
    }
    
    def __init__(
        self,
        provider: str = None,
        secret_name: str = "bynoemie/api-keys",
        **kwargs
    ):
        self.secret_name = secret_name
        self.provider = self._detect_provider(provider)
        self._provider_instance = self._create_provider(**kwargs)
        self._secrets: Dict[str, str] = {}
    
    def _detect_provider(self, provider: str = None) -> str:
        """Auto-detect cloud provider based on environment"""
        if provider:
            return provider.lower()
        
        # Check for cloud-specific environment variables
        if os.getenv("AWS_EXECUTION_ENV") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            return "aws"
        if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID"):
            return "gcp"
        if os.getenv("AZURE_VAULT_URL") or os.getenv("AZURE_CLIENT_ID"):
            return "azure"
        if os.getenv("VAULT_ADDR"):
            return "vault"
        
        # Default to local
        return "local"
    
    def _create_provider(self, **kwargs) -> SecretsProvider:
        """Create the appropriate provider instance"""
        provider_class = self.PROVIDER_MAP.get(self.provider)
        
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        return provider_class(**kwargs)
    
    def load(self) -> Dict[str, str]:
        """Load secrets from provider"""
        try:
            self._secrets = self._provider_instance.get_secrets(self.secret_name)
            logger.info(f"Loaded {len(self._secrets)} secrets from {self.provider}")
            return self._secrets
        except Exception as e:
            logger.warning(f"Failed to load from {self.provider}: {e}")
            
            # Fallback to local if not already
            if self.provider != "local":
                logger.info("Falling back to local .env")
                self._provider_instance = LocalEnvProvider()
                self._secrets = self._provider_instance.get_secrets()
                return self._secrets
            
            raise
    
    def load_to_env(self, overwrite: bool = False):
        """Load secrets and set as environment variables"""
        if not self._secrets:
            self.load()
        
        for key, value in self._secrets.items():
            if overwrite or key not in os.environ:
                os.environ[key] = value
                logger.debug(f"Set environment variable: {key}")
    
    def get(self, key: str, default: str = None) -> Optional[str]:
        """Get a specific secret"""
        if not self._secrets:
            self.load()
        
        return self._secrets.get(key) or os.getenv(key, default)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_secrets_manager: Optional[SecretsManager] = None


def load_secrets(
    provider: str = None,
    secret_name: str = "bynoemie/api-keys",
    **kwargs
) -> Dict[str, str]:
    """
    Load secrets from cloud provider or local .env
    
    Args:
        provider: Cloud provider (aws, gcp, azure, vault, local)
        secret_name: Name/path of the secret
        **kwargs: Provider-specific arguments
        
    Returns:
        Dictionary of secrets
    """
    global _secrets_manager
    
    _secrets_manager = SecretsManager(
        provider=provider,
        secret_name=secret_name,
        **kwargs
    )
    
    _secrets_manager.load_to_env()
    return _secrets_manager._secrets


def get_secret(key: str, default: str = None) -> Optional[str]:
    """Get a specific secret value"""
    global _secrets_manager
    
    if _secrets_manager is None:
        load_secrets()
    
    return _secrets_manager.get(key, default)


# =============================================================================
# AUTO-LOAD ON IMPORT (if in cloud environment)
# =============================================================================

def _auto_load():
    """Automatically load secrets if in cloud environment"""
    # Check if we're in a cloud environment
    is_cloud = any([
        os.getenv("AWS_EXECUTION_ENV"),
        os.getenv("GOOGLE_CLOUD_PROJECT"),
        os.getenv("AZURE_VAULT_URL"),
        os.getenv("VAULT_ADDR"),
    ])
    
    if is_cloud:
        try:
            load_secrets()
        except Exception as e:
            logger.warning(f"Auto-load secrets failed: {e}")

# Uncomment to enable auto-loading
# _auto_load()
