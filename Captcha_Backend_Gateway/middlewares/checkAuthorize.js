require("dotenv").config();
const { responseServerError, responseInValid, responseSuccessWithData } = require("../helpers/ResponseRequest");
const Users = require("../src/users/user.model");

async function checkAuthorize(id, res) {
  const user = await Users.findOne({
    userId: id
  });

  if (user.type !== 1) {
    return responseServerError({ res, err: "Bạn không có quyền thực hiện tác vụ này" })
  }
}

module.exports = { checkAuthorize };
