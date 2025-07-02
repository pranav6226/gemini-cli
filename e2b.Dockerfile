# Use official Node.js 20 image
FROM node:20

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files and install dependencies first (for better caching)
COPY package*.json ./
COPY packages ./packages

RUN npm install

# Copy the rest of the code
COPY . .

# Set environment variable for OpenAI API key (can be overridden at runtime)
ENV OPENAI_API_KEY=""

# Default command: run the CLI
CMD ["npx", "."]
