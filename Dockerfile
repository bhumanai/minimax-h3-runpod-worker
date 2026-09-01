FROM runpod/worker-comfyui:5.8.6-base

ARG COMFYUI_COMMIT=3216c62e9962c3babd28a4dfea6e5aef50b8fe16

RUN git config --global --add safe.directory /comfyui \
    && git -C /comfyui fetch --depth 1 https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_COMMIT}" \
    && git -C /comfyui checkout --detach "${COMFYUI_COMMIT}" \
    && uv pip install -r /comfyui/requirements.txt \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && cd /comfyui \
    && timeout 300 python main.py --quick-test-for-ci --cpu \
    && python -c "import asyncio, nodes; asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False)); assert 'MiniMaxH3ReferenceToVideo' in nodes.NODE_CLASS_MAPPINGS"

COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY download_models.py /download_models.py
COPY bootstrap-h3.sh /bootstrap-h3.sh

RUN chmod +x /bootstrap-h3.sh

CMD ["/bootstrap-h3.sh"]
