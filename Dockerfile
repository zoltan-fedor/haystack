FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y \
    curl  \
    git  \
    pkg-config  \
    cmake \
    libpoppler-cpp-dev  \
    tesseract-ocr  \
    libtesseract-dev  \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz && \
    tar -xvf xpdf-tools-linux-4.03.tar.gz && cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

# install as a package
COPY setup.py setup.cfg pyproject.toml VERSION.txt LICENSE README.md \
    # Haystack code
    haystack \
    # Saved models
    models* \
    # REST API cpde
    rest_api \
    /home/user/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .[docstores,crawler,preprocessing,ocr,ray,rest]
RUN ls /home/user
RUN pip freeze
RUN python3 -c "from haystack.utils.docker import cache_models;cache_models()"

# create folder for /file-upload API endpoint with write permissions, this might be adjusted depending on FILE_UPLOAD_PATH
RUN mkdir -p /home/user/file-upload
RUN chmod 777 /home/user/file-upload


# optional : copy sqlite db if needed for testing
#COPY qa.db /home/user/

# optional: copy data directory containing docs for ingestion
#COPY data /home/user/data

EXPOSE 8000

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app",  "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180"]
