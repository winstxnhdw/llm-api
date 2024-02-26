# llm-api

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![deploy.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/deploy.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/deploy.yml)
[![build.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/build.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/build.yml)
[![formatter.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/formatter.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/formatter.yml)
[![warmer.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/warmer.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/warmer.yml)
[![dependabot.yml](https://github.com/winstxnhdw/llm-api/actions/workflows/dependabot.yml/badge.svg)](https://github.com/winstxnhdw/llm-api/actions/workflows/dependabot.yml)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/winstxnhdw/llm-api)
[![Open a Pull Request](https://huggingface.co/datasets/huggingface/badges/raw/main/open-a-pr-md-dark.svg)](https://github.com/winstxnhdw/llm-api/compare)

A fast CPU-based API for OpenChat 3.5, hosted on Hugging Face Spaces. To achieve faster executions, we are using [CTranslate2](https://github.com/OpenNMT/CTranslate2) as our inference engine.

## Usage

Simply cURL the endpoint like in the following.

```bash
curl -N 'https://winstxnhdw-llm-api.hf.space/api/v1/generate' \
     -H 'Content-Type: application/json' \
     -d \
     '{
         "instruction": "What is the capital of Japan?"
      }'
```

## Development

First, install the required dependencies for your editor with the following.

```bash
poetry install
```

Now, you can access the Swagger UI at [localhost:7860/api/docs](http://localhost:7860/api/docs) after spinning the server up locally with the following.

```bash
docker build -f Dockerfile.build -t llm-api .
docker run --rm -e APP_PORT=7860 -p 7860:7860 llm-api
```
