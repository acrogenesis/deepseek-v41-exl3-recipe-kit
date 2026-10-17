FROM localhost/glm53-sm120:20261001
ENV CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
COPY work/exllamav3 /opt/deepseek-exllamav3
WORKDIR /opt/deepseek-exllamav3
RUN python -m pip install --no-build-isolation --no-deps .
RUN python -m pip install 'marisa-trie==1.3.1' 'llguidance==1.7.5'
WORKDIR /recipe
ENTRYPOINT ["python"]
