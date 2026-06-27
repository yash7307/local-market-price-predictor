# Deploying PriceIQ on AWS

This project is fully container-ready with two Docker images, a one-command deploy
script, and optional CI/CD via GitHub Actions.

| Component | Port | Health Check | Dockerfile |
|-----------|------|--------------|------------|
| Streamlit Dashboard | `8501` | `/_stcore/health` | `Dockerfile` |
| FastAPI API | `8000` | `/health` | `Dockerfile.api` |

The dashboard auto-trains the model on first launch (~10 sec), so you **do not**
need to upload `data/processed/demand_model.pkl`.

---

## Prerequisites

1. **Docker Desktop** — running and logged in.
2. **AWS CLI v2** — installed and configured:

```powershell
aws configure
# Enter: Access Key, Secret Key, Region (ap-south-1), Output (json)
```

3. **Permissions** — your IAM user needs: `ecr:*`, `apprunner:*`, `iam:PassRole`.

---

## Option 1 — App Runner (Recommended for Demo / Portfolio)

> Fully managed. AWS handles HTTPS, scaling, and restarts. ~$15-30/month.

### One-Command Deploy

```powershell
# Dashboard only
.\deploy-apprunner.ps1

# Dashboard + API
.\deploy-apprunner.ps1 -IncludeApi

# Custom region
.\deploy-apprunner.ps1 -Region us-east-1 -IncludeApi
```

The script will:
1. Create ECR repositories (if they don't exist)
2. Build and push Docker images
3. Print the image URIs and next steps

### Create the App Runner Service (Console)

After the script completes:

1. Open **AWS Console → App Runner → Create service**.
2. Source: **Container registry → Amazon ECR**.
3. Image URI: use the URI printed by the script.
4. Deployment trigger: **Manual** (or Automatic for auto-deploy on ECR push).
5. Port: `8501` for dashboard, `8000` for API.
6. CPU / Memory: `1 vCPU / 2 GB`.
7. Health check path: `/_stcore/health` (dashboard) or `/health` (API).
8. Add environment variable: `ALLOWED_ORIGINS` = your dashboard's App Runner URL.
9. Create and wait ~3 minutes for the service URL.

### Post-Deploy

After your dashboard is live at `https://abc123.ap-south-1.awsapprunner.com`:

- Set `ALLOWED_ORIGINS=https://abc123.ap-south-1.awsapprunner.com` on the **API** service.
- API docs will be at `https://YOUR_API_URL/docs`.

---

## Option 2 — EC2 + Docker Compose (Cheapest)

> Run both services on one server. ~$8-15/month on t3.small. You manage updates and HTTPS.

### Launch EC2

1. Launch an **Ubuntu 22.04** EC2 instance (`t3.small` or `t3.micro`).
2. Security group inbound rules: ports `22`, `8501`, and `8000`.
3. SSH in:

```bash
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
```

### Install & Run

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker ubuntu
newgrp docker

# Clone and start
git clone https://github.com/yash7307/local-market-price-predictor.git
cd local-market-price-predictor

# (Optional) Create production env
cp .env.production .env
nano .env   # Set ALLOWED_ORIGINS=http://EC2_PUBLIC_IP:8501

# Start services
docker compose up -d --build
```

### Access

| Service | URL |
|---------|-----|
| Dashboard | `http://EC2_PUBLIC_IP:8501` |
| API Docs | `http://EC2_PUBLIC_IP:8000/docs` |

### Update (after git push)

```bash
cd local-market-price-predictor
git pull
docker compose up -d --build
```

---

## CI/CD with GitHub Actions

The workflow at `.github/workflows/deploy-aws.yml` auto-deploys on every push to `main`.

### Setup

1. Go to **GitHub → Settings → Secrets and variables → Actions**.
2. Add these **Repository Secrets**:

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | Your IAM access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key |
| `AWS_REGION` | `ap-south-1` (or your region) |
| `AWS_ACCOUNT_ID` | Your 12-digit account ID |
| `DEPLOY_API` | `true` (only if deploying API too) |

3. Push to `main` — the workflow builds images, pushes to ECR, and triggers App Runner re-deploy.

---

## Environment Variables

| Variable | Where | Default | Purpose |
|----------|-------|---------|---------|
| `ALLOWED_ORIGINS` | API container | `*` | Comma-separated CORS origins |
| `PYTHONUNBUFFERED` | Both | `1` | Real-time log output |

Set these as:
- **App Runner**: Configuration → Environment variables
- **EC2**: In `.env.production` (auto-loaded by docker-compose)

---

## Cost Estimates

| Strategy | Monthly Estimate | What You Get |
|----------|-----------------|--------------|
| App Runner (1 service) | ~$15-25 | Dashboard with HTTPS, auto-scaling |
| App Runner (2 services) | ~$30-50 | Dashboard + API, both managed |
| EC2 t3.micro | ~$8-10 | Both services, manual management |
| EC2 t3.small | ~$12-18 | Both services, more headroom |

> ECR storage is ~$0.10/GB/month. Negligible for this project.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| First request is slow | Normal — model trains on first launch (~10 sec) |
| ECR push denied | Run `aws ecr get-login-password ...` again (tokens expire in 12h) |
| App Runner shows "unhealthy" | Increase start period to 120s (model training takes time) |
| CORS errors in browser | Set `ALLOWED_ORIGINS` on the API service to your dashboard URL |
| Docker build fails on ARM Mac | Add `--platform linux/amd64` to the build command |

---

## File Reference

| File | Purpose |
|------|---------|
| `Dockerfile` | Streamlit dashboard image |
| `Dockerfile.api` | FastAPI backend image |
| `docker-compose.yml` | Local multi-service setup |
| `.env.production` | Production environment template |
| `deploy-apprunner.ps1` | One-command ECR push script |
| `.github/workflows/deploy-aws.yml` | CI/CD pipeline |
