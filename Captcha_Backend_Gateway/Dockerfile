FROM node:14-alpine
RUN mkdir -p /home/node/app/node_modules && chown -R node:node /home/node/app
RUN npm install -g nodemon
RUN npm install -g socket.io
RUN npm install -g multer
WORKDIR /home/node/app
COPY package*.json ./
USER node
RUN npm install
COPY --chown=node:node . .
EXPOSE 6790

CMD [ "node", "docker-entry.js" ]
