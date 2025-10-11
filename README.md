# llm-api

![API](https://img.shields.io/endpoint?url=https%3A%2F%2Fwinstxnhdw-llm-api.hf.space%2Fapi%2Fhealth&logo=huggingface&label=llm-api&labelColor=%230B0F19&color=brightgreen&link=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fwinstxnhdw%2Fllm-api)
[![build.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/main.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/main.yml)
[![deploy.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/deploy.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/deploy.yml)
[![formatter.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/formatter.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/formatter.yml)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/winstxnhdw/llm-api)
[![Open a Pull Request](https://huggingface.co/datasets/huggingface/badges/raw/main/open-a-pr-md-dark.svg)](https://github.com/winstxnhdw/llm-api/compare)

A fast CPU-based API for Qwen 2.5, hosted on Hugging Face Spaces. To achieve faster executions, we are using [CTranslate2](https://github.com/OpenNMT/CTranslate2) as our inference engine.

## Usage

Simply cURL the endpoint like in the following.

```bash
curl -N 'https://winstxnhdw-llm-api.hf.space/api/v1/chat/stream' \
     -H 'Content-Type: application/json' \
     -d \
     '{
         "messages": [
             {
                 "role": "user",
                 "content": "What is the capital of France?"
             }
         ]
      }'
```

## Development

There are a few ways to run `llm-api` locally for development.

### Local

If you spin up the server using `uv`, you may access the Swagger UI at [localhost:49494/schema/swagger](http://localhost:49494/schema/swagger).

```bash
uv run llm-api
```

### Docker

You can access the Swagger UI at [localhost:7860/schema/swagger](http://localhost:7860/schema/swagger) after spinning the server up with Docker.

```bash
docker build -f Dockerfile.build -t llm-api .
docker run --rm --init -e SERVER_PORT=7860 -p 7860:7860 llm-api
```

You can enable CUDA support by building the image with the following `--build-arg` flag.

```bash
docker build -f Dockerfile.build -t llm-api --build-arg USE_CUDA=1 .
docker run --rm --init --gpus all -e SERVER_PORT=7860 -p 7860:7860 llm-api
```
