param(
    [string]$Region = "ap-south-1",
    [switch]$IncludeApi
)

$ErrorActionPreference = "Stop"

Write-Host "Resolving AWS Account..."
$AccountId = (aws sts get-caller-identity --query Account --output text)
if (-not $AccountId) {
    Write-Error "Failed to get AWS account ID. Run 'aws configure' first."
    exit 1
}
$ECR_BASE = "$AccountId.dkr.ecr.$Region.amazonaws.com"
Write-Host "Account: $AccountId | Region: $Region"

Write-Host "Logging Docker into ECR..."
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ECR_BASE
if ($LASTEXITCODE -ne 0) { Write-Error "ECR login failed."; exit 1 }

function Deploy-Image {
    param(
        [string]$RepoName,
        [string]$DockerfilePath,
        [string]$Port
    )

    Write-Host "Deploying $RepoName..."

    aws ecr create-repository --repository-name $RepoName --region $Region 2>$null

    Write-Host "Building Docker image..."
    docker build -f $DockerfilePath -t "${RepoName}:latest" .
    if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed for $RepoName."; exit 1 }

    $ImageUri = "${ECR_BASE}/${RepoName}:latest"
    docker tag "${RepoName}:latest" $ImageUri

    Write-Host "Pushing to ECR..."
    docker push $ImageUri
    if ($LASTEXITCODE -ne 0) { Write-Error "Docker push failed for $RepoName."; exit 1 }

    Write-Host "Pushed: $ImageUri"
    Write-Host "Port: $Port"
}

Deploy-Image -RepoName "priceiq-dashboard" -DockerfilePath "Dockerfile" -Port "8501"

if ($IncludeApi) {
    Deploy-Image -RepoName "priceiq-api" -DockerfilePath "Dockerfile.api" -Port "8000"
}

Write-Host "Images pushed to ECR successfully!"
