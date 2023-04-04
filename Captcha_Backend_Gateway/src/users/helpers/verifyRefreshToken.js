const jwt = require("jsonwebtoken");
require("dotenv").config();
const optionsRefresh = {
  expiresIn: "30d",
};
async function verifyRefreshToken(refreshToken) {
  try {
    result = jwt.verify(refreshToken, process.env.REFRESH_SECRET, optionsRefresh);
    if (result) {
      return {
        status: true,
        message: "Refresh token valid",
        tokenDetails: result,
      }
    }
    else {
      return {
        status: false,
        message: "Invalid refresh token"
      }
    }
  } catch (error) {
    return { status: false, error: error };
  }
}

module.exports = { verifyRefreshToken };
