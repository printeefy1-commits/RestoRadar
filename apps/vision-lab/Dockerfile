FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/cache MODEL_DIR=/models VIDEO_SAMPLE_PATH=/recordings/restaurant-demo.mp4
RUN --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && if [ -s /run/secrets/build_ca ]; then \
      apt-get -o Acquire::https::CaInfo=/run/secrets/build_ca update \
      && apt-get -o Acquire::https::CaInfo=/run/secrets/build_ca install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0; \
    else apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0; fi \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
FROM base AS detector-export
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics YOLO_AUTOINSTALL=false
COPY apps/vision-lab/requirements-export.txt ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu \
    && pip install --no-cache-dir -r requirements-export.txt
COPY apps/vision-lab/export_detector.py apps/vision-lab/models.py ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export SSL_CERT_FILE=/run/secrets/build_ca; fi \
    && python export_detector.py /export

FROM detector-export AS yolo26-export
COPY apps/vision-lab/requirements-yolo26-export.txt ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
    && pip install --no-cache-dir -r requirements-yolo26-export.txt
COPY apps/vision-lab/export_yolo26.py apps/vision-lab/yolo26_models.py ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export SSL_CERT_FILE=/run/secrets/build_ca; fi \
    && python export_yolo26.py /export26

FROM base AS runtime
COPY apps/vision-lab/requirements.txt ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then \
      PIP_CERT=/run/secrets/build_ca pip install --no-cache-dir -r requirements.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi
COPY --from=detector-export /export/yolo11l.onnx /export/detector.json /export/ULTRALYTICS-LICENSE /opt/restoradar-detector/
COPY --from=yolo26-export /export26/yolo26n-pose.onnx /export26/yolo26n-pose.json /export26/YOLO26-LICENSE /opt/restoradar-detector/
COPY --chown=10001:10001 apps/vision-lab/*.py ./
RUN useradd --uid 10001 --create-home vision && mkdir /models && chown vision:vision /models
USER vision
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "service.py"]
