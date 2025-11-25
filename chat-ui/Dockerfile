# frontend dockerfile
FROM node:22-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tailwind.config.js ./
COPY postcss.config.js ./
COPY pnpm-lock.yaml  ./
COPY tsconfig.json ./
COPY public/ ./public/
COPY src/  ./src/

# Install dependencies
RUN npm install

# Build the React app
RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]