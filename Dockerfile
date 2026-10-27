FROM mambaorg/micromamba:2.3.2
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ENV R_LIBS_USER=/app/.r-library OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN python -c "from lab.runtime import prepare_runtime; s=prepare_runtime(); print(s); assert s['available']"
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["/opt/conda/bin/python", "scripts/healthcheck.py"]
CMD ["python", "scripts/serve.py"]
