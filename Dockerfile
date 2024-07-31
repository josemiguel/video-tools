# Use the official AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Install necessary dependencies using yum
RUN dnf install -y \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    mesa-libGLU \
    libXrandr \
    libXcursor \
    libXinerama \
    libXi \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel

# Copy the requirements file first and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the function code last to leverage Docker caching
COPY app.py ${LAMBDA_TASK_ROOT}/app.py

# Set the CMD to your handler (file.function)
CMD ["app.handler"]
