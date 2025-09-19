# Usa a imagem base da Hugging Face
FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Cria um ambiente virtual e instala todas as dependências do projeto nele
# O pacote "Brotli" foi removido pois estava causando o erro de "no matching distribution"
# O uso de 'line continuation' (\) torna o comando mais legível
RUN python3.12 -m pip install --no-cache-dir \
    aiofiles==24.1.0 \
    annotated-types==0.7.0 \
    anyio==4.10.0 \
    attrs==25.3.0 \
    certifi==2025.8.3 \
    charset-normalizer==3.4.3 \
    click==8.3.0 \
    colorama==0.4.6 \
    coloredlogs==15.0.1 \
    einops==0.8.1 \
    executing==2.2.1 \
    fastapi==0.116.2 \
    ffmpy==0.6.1 \
    filelock==3.19.1 \
    flatbuffers==25.2.10 \
    fpsample==0.3.3 \
    fsspec==2025.9.0 \
    gradio==5.46.1 \
    gradio_client==1.13.1 \
    groovy==0.1.2 \
    h11==0.16.0 \
    hf-xet==1.1.10 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface-hub==0.35.0 \
    humanfriendly==10.0 \
    idna==3.10 \
    imageio==2.37.0 \
    Jinja2==3.1.6 \
    jsonschema==4.25.1 \
    jsonschema-specifications==2025.9.1 \
    kiui==0.2.18 \
    lazy_loader==0.4 \
    llvmlite==0.45.0 \
    markdown-it-py==4.0.0 \
    MarkupSafe==3.0.2 \
    mdurl==0.1.2 \
    mpmath==1.3.0 \
    #msvc_runtime==14.44.35112 \
    networkx==3.5 \
    ninja==1.13.0 \
    numba==0.62.0 \
    numpy==2.2.6 \
    objprint==0.3.0 \
    onnxruntime==1.22.1 \
    opencv-python==4.12.0.88 \
    opencv-python-headless==4.12.0.88 \
    orjson==3.11.3 \
    packaging==25.0 \
    pandas==2.3.2 \
    pillow==11.3.0 \
    platformdirs==4.4.0 \
    pooch==1.8.2 \
    protobuf==6.32.1 \
    pydantic==2.11.9 \
    pydantic_core==2.33.2 \
    pydub==0.25.1 \
    Pygments==2.19.2 \
    PyMatting==1.1.14 \
    PyMCubes==0.1.6 \
    pymeshlab==2025.7 \
    pyreadline3==3.5.4 \
    python-dateutil==2.9.0.post0 \
    python-multipart==0.0.20 \
    pytz==2025.2 \
    PyYAML==6.0.2 \
    referencing==0.36.2 \
    regex==2025.9.18 \
    rembg==2.0.67 \
    requests==2.32.5 \
    rich==14.1.0 \
    rpds-py==0.27.1 \
    ruff==0.13.1 \
    safehttpx==0.1.6 \
    safetensors==0.6.2 \
    scikit-image==0.25.2 \
    scipy==1.16.2 \
    semantic-version==2.10.0 \
    setuptools==80.9.0 \
    shellingham==1.5.4 \
    six==1.17.0 \
    sniffio==1.3.1 \
    starlette==0.48.0 \
    sympy==1.14.0 \
    tifffile==2025.9.9 \
    tokenizers==0.22.1 \
    tomlkit==0.13.3 \
    tqdm==4.67.1 \
    transformers==4.56.2 \
    trimesh==4.8.2 \
    typer==0.18.0 \
    typing-inspection==0.4.1 \
    typing_extensions==4.15.0 \
    tzdata==2025.2 \
    urllib3==2.5.0 \
    uvicorn==0.35.0 \
    varname==0.15.0 \
    websockets==15.0.1 \
    torch==2.8.0+cu126 \
    torchaudio==2.8.0+cu126 \
    torchvision==0.23.0+cu126 \
     --extra-index-url https://download.pytorch.org/whl/cu126

# Define a porta que o container vai expor
COPY . .
EXPOSE 7860

# Comando para iniciar a aplicação a partir do ambiente virtual
CMD ["python", "app.py"]
