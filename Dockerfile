# Define the build stage
FROM node:20-alpine as build

WORKDIR /usr/src/app

# Copy package.json and other dependency-related files
COPY package*.json ./

# Install all dependencies
RUN npm install

# Copy the rest of your application code
COPY . .

# Build your application
RUN npm run build

# Setup CodeQL
RUN npm run codeql-setup

# Define the production stage
FROM node:20-alpine as production

ARG NODE_ENV=production
ENV NODE_ENV=${NODE_ENV}

WORKDIR /usr/src/app

# Copy the built artifacts from the build stage
COPY --from=build /usr/src/app/dist ./dist
COPY package*.json ./

# Install only production dependencies
RUN npm install --only=production

# Expose the port your app runs on
EXPOSE 8000

# Command to run your app
CMD ["node", "dist/main"]