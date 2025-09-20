# Use uma imagem de base mais limpa e específica
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL name="partpacker" maintainer="partpacker"

# Define o locale para evitar warnings.
ENV LANG=C.UTF-8

# Atualiza listas de pacotes e instala todas as dependências do sistema em uma única camada.
# Isso inclui bibliotecas de gráficos (libgl, libegl), GTK/GLib (libglib2.0-0, libsm6),
# e ferramentas de build (git, wget, build-essential).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libgl1 \
    libgles2 \
    libegl1 \
    libxext6 \
    libxrender-dev \
    libxi6 \
    libgconf-2-4 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Cria o diretório de trabalho.
WORKDIR /workspace/PartPacker

# Clona o repositório antes de instalar as dependências Python
# para que o requirements.txt esteja disponível.
RUN git clone https://github.com/NVlabs/PartPacker.git .

# Limpa o arquivo de requisitos para evitar erros com pip
RUN sed -i 's/ --no-build-isolation//g' requirements.txt

# Instala as dependências Python e outras bibliotecas de forma otimizada.
# O torch é instalado com a URL correta e sem Conda.
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir \
    -r requirements.txt \
    transformers \
    # Remove as dependências de build após a instalação
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Cria o diretório para modelos e baixa os arquivos.
RUN mkdir -p pretrained && \
    wget -P pretrained https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt && \
    wget -P pretrained https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt

# Modifica o arquivo app.py para habilitar o modo de compartilhamento
RUN sed -i 's/block\.launch()/block.launch(share=True)/g' app.py

# Expõe a porta e define o comando padrão
EXPOSE 7860
CMD ["python3", "app.py"]
