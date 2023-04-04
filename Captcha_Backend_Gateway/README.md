
<!-- ABOUT THE ATTT BACKEND REST API  -->

### What's inside this repo?
- Rest API

### Built With

- [Node.js]() - JavaScript runtime built on Chrome's V8 JavaScript engine.
- [Express.js]() - Minimal and flexible Node.js web application framework
- [MongoDB]() - Cross-platform document-oriented database program

<!-- GETTING STARTED -->

## Getting Started

- Download source code or clone the repository
- Default API port : 6789 set in Dockerfile
- Default MongoDB port : 27017 set in docker-compose.yml

### Prerequisites

To run this project, you'll need to have the following installed:

- Node.js : [https://nodejs.org](https://nodejs.org)

- npm :
  ```sh
  npm install npm@latest -g
  ```
- Docker Compose: [https://huongdan.azdigi.com/cai-dat-va-su-dung-docker-compose-ubuntu-22-04/]
- MongoDB : [https://mongodb.com](https://mongodb.com) <br>

> You can also use MongoDB Atlas if you prefer.
> <br>

### Installation

1. Start MongoDB container (use `sudo` if required) :
   ```sh
   docker-compose up -d
   ```
2. To shutdown database without remove the container (use `sudo` if required) :

   ```sh
   docker-compose stop
   ```

3. To shutdown database and remove the container (use `sudo` if required) :

   ```sh
   docker-compose down
   ```
4. Create `.env` file and configure :
   ```JS
   MONGO_URI = <MONGODB_URL>
   JWT_SECRET = <SOME_LONG_SECURE_SECRET>
   ```
5. Start the server :
   ```sh
   npm start
   ```
