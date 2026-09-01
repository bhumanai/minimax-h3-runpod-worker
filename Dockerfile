FROM runpod/worker-comfyui:5.8.6-base

COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY download_models.py /download_models.py
COPY bootstrap-h3.sh /bootstrap-h3.sh

RUN chmod +x /bootstrap-h3.sh

CMD ["/bootstrap-h3.sh"]
