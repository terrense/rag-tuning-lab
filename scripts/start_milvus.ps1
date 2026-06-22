param(
    [string]$ComposeFile = "infra/milvus/docker-compose.yml"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found on PATH. Start Docker Desktop and try again."
}

docker compose -f $ComposeFile up -d
docker ps --filter "name=rag-lab-milvus" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
