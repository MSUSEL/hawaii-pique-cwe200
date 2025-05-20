# download java
FROM alpine/curl:8.12.1 AS java_download
ARG JAVA_VERSION=21
ENV JAVA_RELEASE="https://github.com/adoptium/temurin$JAVA_VERSION-binaries/releases/download"

WORKDIR /java
RUN if [ "$JAVA_VERSION" = "8" ]; then \
        curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u372-b07/OpenJDK8U-jdk_x64_linux_hotspot_8u372b07.tar.gz; \
    elif [ "$JAVA_VERSION" = "11" ]; then \
        curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.20+8/OpenJDK11U-jdk_x64_linux_hotspot_11.0.20_8.tar.gz; \
    elif [ "$JAVA_VERSION" = "17" ]; then \
        curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.8+7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.8_7.tar.gz; \
    elif [ "$JAVA_VERSION" = "21" ]; then \
        curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.2+13/OpenJDK21U-jdk_x64_linux_hotspot_21.0.2_13.tar.gz; \
    else \
        echo "Unsupported Java version $JAVA_VERSION"; exit 1; \
    fi \
    && mkdir -p /java/jdk \
    && tar xzf /tmp/openjdk.tar.gz -C /java/jdk --strip-components=1

# dowload codeql
FROM alpine/curl:8.12.1 AS codeql_download
ARG CODEQL_VERSION=2.20.3

WORKDIR /codeql
RUN apk add --no-cache unzip \
    && curl -SsLo /tmp/codeql.zip "https://github.com/github/codeql-cli-binaries/releases/download/v$CODEQL_VERSION/codeql-linux64.zip" \
    && unzip /tmp/codeql.zip


# Use the official Node.js image as the base image
FROM node:20.11.1-bullseye-slim AS runtime
# Expose the ports for frontend and backend
EXPOSE 4200 5400
# Set environment variables for Java, Maven, and Gradle
ARG JAVA_VERSION=21 
ENV JAVA_VERSION=${JAVA_VERSION}
ENV JAVA_HOME=/usr/local/java/jdk${JAVA_VERSION}
ENV MAVEN_HOME=/usr/local/maven
ENV GRADLE_HOME=/usr/local/gradle
ENV CODEQL_HOME=/usr/local/bin/codeql
ENV PATH="${JAVA_HOME}/bin:${MAVEN_HOME}/bin:${GRADLE_HOME}/bin:${CODEQL_HOME}:${PATH}"

# install dependencies
RUN apt update && apt install -y \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    gradle \
    git \
    git-lfs \
    wget \
    ant \
    && git lfs install \
    && ln -s $(which python3) /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Maven 3.9.9 (This is done separately because Apt does not have 3.9.9)
RUN curl -fsSL https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz \
    | tar -xz -C /usr/local/ \
    && mv /usr/local/apache-maven-3.9.9 /usr/local/maven

# Install 'concurrently' globally to run multiple scripts
RUN npm install -g concurrently

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy root package files
COPY package*.json ./

# Copy the rest of the application code
# COPY backend/ /app/backend/
# COPY frontend/ /app/frontend/
# COPY testing/ /app/testing/
# COPY Dockerfile /app/Dockerfile
# COPY .devcontainer/ /app/.devcontainer/
COPY . /app

# Build the frontend/backend via root build script
RUN npm run build && npm cache clean --force

# Setup CodeQL for the application
COPY --from=codeql_download /codeql/codeql/ $CODEQL_HOME/
RUN npm run codeql-setup

# copy java download
COPY --from=java_download /java/jdk $JAVA_HOME

# Start both frontend and backend servers
ENTRYPOINT ["npm", "start"]