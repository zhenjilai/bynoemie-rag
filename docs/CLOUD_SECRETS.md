# Cloud API Key Management Guide

## 1. Cloud Provider Secret Management (Recommended)

### AWS Secrets Manager
```python
import boto3
import json

def get_secret_aws(secret_name: str, region: str = "ap-southeast-1") -> dict:
    """Retrieve secrets from AWS Secrets Manager"""
    client = boto3.client('secretsmanager', region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret_aws("bynoemie/api-keys")
os.environ["GROQ_API_KEY"] = secrets["GROQ_API_KEY"]
```

### Google Cloud Secret Manager
```python
from google.cloud import secretmanager

def get_secret_gcp(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Retrieve secrets from GCP Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
os.environ["GROQ_API_KEY"] = get_secret_gcp("bynoemie-project", "groq-api-key")
```

### Azure Key Vault
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

def get_secret_azure(vault_url: str, secret_name: str) -> str:
    """Retrieve secrets from Azure Key Vault"""
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value

# Usage
os.environ["GROQ_API_KEY"] = get_secret_azure(
    "https://bynoemie-vault.vault.azure.net/", 
    "groq-api-key"
)
```

## 2. Kubernetes Secrets
```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: bynoemie-api-keys
type: Opaque
stringData:
  GROQ_API_KEY: "gsk_xxxx"
  OPENAI_API_KEY: "sk-xxxx"
  LANGCHAIN_API_KEY: "ls__xxxx"
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: bynoemie-rag
        envFrom:
        - secretRef:
            name: bynoemie-api-keys
```

## 3. Docker Secrets (Docker Swarm)
```yaml
# docker-compose.yml
version: '3.8'
services:
  bynoemie-rag:
    image: bynoemie-rag:latest
    secrets:
      - groq_api_key
      - openai_api_key
    environment:
      - GROQ_API_KEY_FILE=/run/secrets/groq_api_key

secrets:
  groq_api_key:
    external: true
  openai_api_key:
    external: true
```

## 4. HashiCorp Vault (Enterprise)
```python
import hvac

def get_secret_vault(path: str) -> dict:
    """Retrieve secrets from HashiCorp Vault"""
    client = hvac.Client(url='https://vault.example.com')
    client.token = os.getenv('VAULT_TOKEN')
    secret = client.secrets.kv.v2.read_secret_version(path=path)
    return secret['data']['data']

# Usage
secrets = get_secret_vault("bynoemie/api-keys")
os.environ["GROQ_API_KEY"] = secrets["groq_api_key"]
```

## Comparison Table

| Method | Best For | Cost | Complexity |
|--------|----------|------|------------|
| AWS Secrets Manager | AWS deployments | $0.40/secret/month | Low |
| GCP Secret Manager | GCP deployments | $0.06/10K access | Low |
| Azure Key Vault | Azure deployments | $0.03/10K ops | Low |
| Kubernetes Secrets | K8s clusters | Free | Medium |
| HashiCorp Vault | Multi-cloud/Enterprise | $$$ | High |
| Environment Variables | Simple deployments | Free | Low |

## Recommendation by Deployment Type

| Deployment | Recommended Method |
|------------|-------------------|
| AWS ECS/Lambda | AWS Secrets Manager |
| Google Cloud Run | GCP Secret Manager |
| Azure Container Apps | Azure Key Vault |
| Kubernetes (any cloud) | K8s Secrets + External Secrets Operator |
| Docker Compose (small) | Environment variables in .env |
| Heroku | Config Vars (built-in) |
| Railway/Render | Platform secrets (built-in) |
