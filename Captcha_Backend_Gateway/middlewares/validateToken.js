const jwt = require("jsonwebtoken");
require("dotenv").config();

const User = require("../src/users/user.model");

async function validateToken(req, res, next) {
  const authorizationHeader = req.headers.authorization;
  let result;
  if (!authorizationHeader)
    return res.status(401).json({
      status: false,
      message: "Access token is missing",
    });

  const token = req.headers.authorization.split(" ")[1]; // Bearer <token>
  const options = {
    expiresIn: process.env.EXPIRE_TOKEN || "1m",
  };
  try {
    result = jwt.verify(token, process.env.JWT_SECRET, options);
    let user = await User.findOne({
      userId: result.id,
    });
    if (!user) {
      result = {
        status: false,
        message: `Authorization error`,
      };
      return res.status(403).json(result);
    }

    if (!user.genToken === result.genToken) {
      result = {
        status: false,
        message: `Invalid token`,
      };

      return res.status(401).json(result);
    }

    req.decoded = result;
    next();
  } catch (err) {
    if (err.name === "TokenExpiredError") {
      result = {
        status: false,
        message: `TokenExpired`,
      };
    } else {
      result = {
        status: false,
        message: `Authentication error`,
      };
    }
    return res.status(403).json(result);
  }
}

module.exports = { validateToken };
