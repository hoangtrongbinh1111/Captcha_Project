const jwt = require("jsonwebtoken");
require("dotenv").config();

const options = {
  expiresIn: process.env.EXPIRE_TOKEN || "1m",
};
const optionsRefresh = {
  expiresIn: process.env.EXPIRE_REFRESH_TOKEN || "30d",
};

async function generateJwt(payload) {
  try {
    const token = await jwt.sign(payload, process.env.JWT_SECRET, options);
    const refreshToken = await jwt.sign(payload, process.env.REFRESH_SECRET, optionsRefresh);
    return { status: true, token: token, refreshToken: refreshToken };
  } catch (error) {
    return { status: false, message: error };
  }
}

async function generateOnlyJwt(payload) {
  try {
    const token = await jwt.sign(payload, process.env.JWT_SECRET, options);
    return { status: true, token: token };
  } catch (error) {
    return { status: false, message: error };
  }
}

module.exports = { generateJwt, generateOnlyJwt };
