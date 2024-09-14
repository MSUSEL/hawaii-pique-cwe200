# Use the official Node.js image as the base image
FROM node:20.11.1-bullseye-slim

# Set build arguments for Java and Maven versions
ARG JAVA_VERSION=8
ARG MAVEN_VERSION=3.9.5

# Set environment variables for Java and Maven
ENV JAVA_HOME=/usr/local/java
ENV MAVEN_HOME=/usr/local/maven
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

# Install dependencies for building Python and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gnupg \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12.2 from source
ENV PYTHON_VERSION=3.12.2
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
    && tar xzf Python-$PYTHON_VERSION.tgz \
    && cd Python-$PYTHON_VERSION \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-$PYTHON_VERSION.tgz Python-$PYTHON_VERSION \
    && ln -s /usr/local/bin/python3.12 /usr/local/bin/python

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.12 get-pip.py && rm get-pip.py

# Copy requirements.txt
COPY requirements.txt ./

# Install Python dependencies
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Install 'concurrently' globally to run multiple scripts
RUN npm install -g concurrently

# Install Java
RUN mkdir -p $JAVA_HOME \
    && if [ "$JAVA_VERSION" = "8" ]; then \
        wget -O /tmp/openjdk.tar.gz https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u372-b07/OpenJDK8U-jdk_x64_linux_hotspot_8u372b07.tar.gz; \
    elif [ "$JAVA_VERSION" = "11" ]; then \
        wget -O /tmp/openjdk.tar.gz https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.20+8/OpenJDK11U-jdk_x64_linux_hotspot_11.0.20_8.tar.gz; \
    elif [ "$JAVA_VERSION" = "17" ]; then \
        wget -O /tmp/openjdk.tar.gz https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.8+7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.8_7.tar.gz; \
    else \
        echo "Unsupported Java version $JAVA_VERSION"; exit 1; \
    fi \
    && tar -xzf /tmp/openjdk.tar.gz -C $JAVA_HOME --strip-components=1 \
    && rm /tmp/openjdk.tar.gz

# Install Maven
RUN wget -O /tmp/maven.tar.gz https://downloads.apache.org/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz \
    && mkdir -p $MAVEN_HOME \
    && tar -xzf /tmp/maven.tar.gz -C $MAVEN_HOME --strip-components=1 \
    && rm /tmp/maven.tar.gz

# Download and install CodeQL CLI
ENV CODEQL_VERSION=2.17.0
RUN wget -O /tmp/codeql.zip https://github.com/github/codeql-cli-binaries/releases/download/v$CODEQL_VERSION/codeql-linux64.zip \
    && unzip /tmp/codeql.zip -d /opt \
    && rm /tmp/codeql.zip \
    && ln -s /opt/codeql/codeql /usr/local/bin/codeql

# Set the working directory
WORKDIR /app

# Copy root package files
COPY package*.json ./

# Copy backend and frontend package files
COPY backend/package*.json ./backend/
COPY frontend/package*.json ./frontend/

# Install root dependencies if any
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Setup CodeQL for the application
RUN npm run codeql-setup

# Expose the ports for frontend and backend
EXPOSE 4200 5400

# Start both frontend and backend servers
CMD ["npm", "start"]
