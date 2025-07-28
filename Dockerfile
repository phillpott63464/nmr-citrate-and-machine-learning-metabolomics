FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /usr/src/app

# Copy uv configuration files
COPY ./app/pyproject.toml ./app/uv.lock* ./

# Expose port 2718
EXPOSE 2718

CMD ["uv", "run", "marimo", "edit", "--host", "0.0.0.0", "--port", "2718"]

