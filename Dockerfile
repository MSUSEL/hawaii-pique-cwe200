# install java
FROM alpine/curl:8.12.1 AS java_download
ARG JAVA_VERSION=8
ENV JAVA_RELEASE="https://github.com/adoptium/temurin$JAVA_VERSION-binaries/releases/download"

WORKDIR /java
RUN if [ "$JAVA_VERSION" = "8" ]; then \
        curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u372-b07/OpenJDK8U-jdk_x64_linux_hotspot_8u372b07.tar.gz; \
    elif [ "$JAVA_VERSION" = "11" ]; then \
        curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.20+8/OpenJDK11U-jdk_x64_linux_hotspot_11.0.20_8.tar.gz; \
    elif [ "$JAVA_VERSION" = "17" ]; then \
       curl -SsLo /tmp/openjdk.tar.gz https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.8+7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.8_7.tar.gz; \
    else \
        echo "Unsupported Java version $JAVA_VERSION"; exit 1; \
    fi \
    && tar xzf /tmp/openjdk.tar.gz --strip-components=1

# install python
FROM debian:bullseye-slim AS python_build
ARG PYTHON_VERSION=3.12.2

# install prereqs
RUN apt update && apt install curl gcc make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev -y

WORKDIR /python
RUN curl -SsLo /tmp/python.tgz "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz" \
    && tar xzf /tmp/python.tgz \
    && cd Python-$PYTHON_VERSION \
    && ./configure --enable-optimizations \
    && make altinstall

# install maven
FROM alpine/curl:8.12.1 AS maven_download
ARG MAVEN_VERSION=3.9.5

WORKDIR /maven
RUN curl -SsLo /tmp/maven.tar.gz "https://downloads.apache.org/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz" \
    && tar xzf /tmp/maven.tar.gz --strip-components=1


# install gradle
FROM alpine/curl:8.12.1 AS gradle_download
ARG GRADLE_VERSION=8.6

WORKDIR /gradle
RUN curl -SsLo /tmp/gradle.zip "https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip" \
    && unzip /tmp/gradle.zip


# install codeql
FROM alpine/curl:8.12.1 AS codeql_download
ARG CODEQL_VERSION=2.20.3

WORKDIR /codeql
RUN curl -SsLo /tmp/codeql.zip "https://github.com/github/codeql-cli-binaries/releases/download/v$CODEQL_VERSION/codeql-linux64.zip" \
    && unzip /tmp/codeql.zip

#
# Use the official Node.js image as the base image
#
FROM node:20.11.1-bullseye-slim AS runtime
# Expose the ports for frontend and backend
EXPOSE 4200 5400
# Set environment variables for Java, Maven, and Gradle
ENV JAVA_HOME=/usr/local/java
ENV PYTHON_HOME=/usr/local/bin/python
ENV MAVEN_HOME=/usr/local/maven
ENV GRADLE_HOME=/usr/local/gradle
ENV CODEQL_HOME=/usr/local/bin/codeql
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH:$GRADLE_HOME/bin:$PATH

# Install 'concurrently' globally to run multiple scripts
RUN npm install -g concurrently

# Set the working directory
WORKDIR /app

# Copy root package files
COPY package*.json ./

# Copy backend and frontend package files
COPY backend/package*.json ./backend/
COPY frontend/package*.json ./frontend/

# Install root dependencies if any
RUN npm install --omit=dev && npm cache clean --force

# Copy the rest of the application code
COPY . /app

# Build the application
RUN npm run build

# Setup CodeQL for the application
COPY --from=codeql_download /codeql $CODEQL_HOME
RUN npm run codeql-setup

# copy build system and languages
COPY --from=maven_download /maven $MAVEN_HOME
COPY --from=gradle_download /gradle $GRADLE_HOME
COPY --from=java_download /java $JAVA_HOME
COPY --from=python_build /usr/local/bin/python3.12 $PYTHON_HOME
# Install pip for Python 3.12
RUN curl -LSs https://bootstrap.pypa.io/get-pip.py | python get-pip.py

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Start both frontend and backend servers
ENTRYPOINT ["npm", "start"]
