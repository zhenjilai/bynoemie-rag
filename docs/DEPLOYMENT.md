# ByNoemie RAG Chatbot - Production Deployment Guide

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- OpenAI API key

### 1. Local Docker Deployment

```bash
# Clone/extract the project
cd bynoemie_rag_v2

# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Start services
docker-compose up -d

# Access:
# - API: http://localhost:8000
# - UI:  http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

---

## Cloud Deployment Options

### Option 1: Railway (Easiest - Recommended)

**Cost: ~$5-20/month**

1. **Create account** at [railway.app](https://railway.app)

2. **Deploy via GitHub:**
   ```bash
   # Push to GitHub first
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/bynoemie-chatbot.git
   git push -u origin main
   ```

3. **In Railway Dashboard:**
   - Click "New Project" → "Deploy from GitHub"
   - Select your repo
   - Add environment variables:
     - `OPENAI_API_KEY` = your key
     - `PORT` = 8000
   - Railway auto-detects Dockerfile

4. **For Streamlit UI:**
   - Add new service from same repo
   - Set start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

---

### Option 2: Render (Free tier available)

**Cost: Free tier / $7+/month for paid**

1. **Create account** at [render.com](https://render.com)

2. **Create `render.yaml`:**
   ```yaml
   services:
     # API Service
     - type: web
       name: bynoemie-api
       env: docker
       dockerfilePath: ./Dockerfile.production
       dockerContext: .
       envVars:
         - key: OPENAI_API_KEY
           sync: false
         - key: PORT
           value: 8000
       healthCheckPath: /health
   
     # UI Service  
     - type: web
       name: bynoemie-ui
       env: docker
       dockerfilePath: ./Dockerfile.production
       dockerContext: .
       dockerCommand: streamlit run app.py --server.port=$PORT
       envVars:
         - key: OPENAI_API_KEY
           sync: false
   ```

3. **Deploy:**
   - Push to GitHub
   - In Render: "New" → "Blueprint" → Select repo
   - Add API key in dashboard

---

### Option 3: Google Cloud Run (Serverless)

**Cost: Pay per use (~$0-10/month for low traffic)**

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/bynoemie-api

# Deploy API
gcloud run deploy bynoemie-api \
  --image gcr.io/YOUR_PROJECT_ID/bynoemie-api \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=sk-xxx" \
  --memory 1Gi \
  --cpu 1 \
  --port 8000

# Deploy UI (Streamlit)
gcloud run deploy bynoemie-ui \
  --image gcr.io/YOUR_PROJECT_ID/bynoemie-api \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=sk-xxx" \
  --command "streamlit,run,app.py,--server.port=8080" \
  --memory 1Gi \
  --port 8080
```

---

### Option 4: AWS (ECS/Fargate)

**Cost: ~$15-30/month**

```bash
# Install AWS CLI
# Configure: aws configure

# Create ECR repository
aws ecr create-repository --repository-name bynoemie-api

# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

# Build and push
docker build -t bynoemie-api -f Dockerfile.production .
docker tag bynoemie-api:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/bynoemie-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/bynoemie-api:latest

# Use AWS Console or Terraform to create:
# 1. ECS Cluster
# 2. Task Definition
# 3. Service
# 4. Load Balancer
```

---

### Option 5: DigitalOcean App Platform

**Cost: ~$5-12/month**

1. Create account at [digitalocean.com](https://digitalocean.com)

2. Create `app.yaml`:
   ```yaml
   name: bynoemie-chatbot
   services:
     - name: api
       dockerfile_path: Dockerfile.production
       http_port: 8000
       instance_count: 1
       instance_size_slug: basic-xxs
       envs:
         - key: OPENAI_API_KEY
           scope: RUN_TIME
           type: SECRET
       health_check:
         http_path: /health
     
     - name: ui
       dockerfile_path: Dockerfile.production
       http_port: 8501
       run_command: streamlit run app.py --server.port=8501
       instance_count: 1
       instance_size_slug: basic-xxs
       envs:
         - key: OPENAI_API_KEY
           scope: RUN_TIME
           type: SECRET
   ```

3. Deploy via CLI:
   ```bash
   doctl apps create --spec app.yaml
   ```

---

## API Endpoints

Once deployed, your API will have these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send message to chatbot |
| `/search` | POST | Search products |
| `/products` | GET | List all products |
| `/products/{id}` | GET | Get product details |
| `/categories` | GET | Get all categories |
| `/feedback` | POST | Submit feedback |

### Example API Usage

```python
import requests

# Chat with the bot
response = requests.post(
    "https://your-api-url.com/chat",
    json={
        "message": "Show me all heels",
        "session_id": "user-123"
    }
)
print(response.json())

# Search products
response = requests.post(
    "https://your-api-url.com/search",
    json={
        "query": "party dress",
        "limit": 5,
        "category": "Clothing"
    }
)
print(response.json())
```

### JavaScript/Frontend Integration

```javascript
// Chat integration
async function sendMessage(message, sessionId) {
    const response = await fetch('https://your-api-url.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: message,
            session_id: sessionId
        })
    });
    return response.json();
}

// Usage
const result = await sendMessage("Show me dresses", "session-123");
console.log(result.response);  // Bot's reply
console.log(result.products);  // Product recommendations
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `GROQ_API_KEY` | No | Groq API key (optional) |
| `PORT` | No | Server port (default: 8000) |
| `ENV` | No | Environment (production/development) |

---

## Monitoring & Logs

### Docker logs
```bash
docker-compose logs -f api
docker-compose logs -f ui
```

### Cloud platform logs
- **Railway**: Dashboard → Service → Logs
- **Render**: Dashboard → Service → Logs
- **GCP**: Cloud Console → Cloud Run → Logs
- **AWS**: CloudWatch Logs

---

## Scaling

### Horizontal scaling (more instances)
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      replicas: 3
```

### Cloud auto-scaling
Most cloud platforms support auto-scaling based on:
- CPU usage
- Memory usage
- Request count
- Response latency

---

## Security Checklist

- [ ] Use HTTPS (SSL/TLS)
- [ ] Set CORS origins (not `*` in production)
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Add authentication (API keys or OAuth)
- [ ] Regular security updates

---

## Cost Comparison

| Platform | Free Tier | Paid (Basic) | Notes |
|----------|-----------|--------------|-------|
| Railway | $5 credit | $5-20/mo | Easiest setup |
| Render | Yes | $7+/mo | Good free tier |
| GCP Cloud Run | Yes | Pay per use | Best for variable traffic |
| AWS Fargate | No | $15-30/mo | Enterprise features |
| DigitalOcean | No | $5-12/mo | Simple pricing |

---

## Recommended: Railway Deployment

For the fastest path to production:

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Add environment variables
railway variables set OPENAI_API_KEY=sk-xxx

# 5. Deploy
railway up

# Done! Get your URL from the dashboard
```

Your chatbot will be live at: `https://your-project.up.railway.app`
