FROM node:22 as build

RUN git clone https://github.com/dragoscalin33/flight-on-time-ds.git /app
WORKDIR /app/front-end
RUN npm install
RUN npm run build

FROM caddy:latest
WORKDIR /usr/share/caddy
COPY --from=build /app/front-end/dist/index.html ./index.html
COPY --from=build /app/front-end/dist/assets ./assets 
COPY Caddyfile /etc/caddy/Caddyfile

# docker build -t rrbotlab/flight-front:dev-v1 .
