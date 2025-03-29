# Make sure to use this base image
FROM e2bdev/code-interpreter:latest

RUN pip install uv

RUN npm install -g pyright
