const Joi = require("joi");
require("dotenv").config();
const { v4: uuid } = require("uuid");
const { customAlphabet: generate } = require("nanoid");
const { responseServerError, responseInValid, responseSuccessWithData, responseSuccess } = require("../../helpers/ResponseRequest");

const { generateJwt, generateOnlyJwt } = require("./helpers/generateJwt");
const { verifyRefreshToken } = require("./helpers/verifyRefreshToken");
const User = require("./user.model");

//Validate user schema
const userSchema = Joi.object().keys({
  fullname: Joi.string()
    .min(3)
    .max(30)
    .required(),
  phoneNumber: Joi.string().min(8).max(10).pattern(/^[0-9]+$/).required(),
  email: Joi.string().email({ minDomainSegments: 2 }),
  username: Joi.string().required().min(6),
  password: Joi.string().required().min(4),
  confirmPassword: Joi.string().valid(Joi.ref("password")).required()
});

const EditUserSchema = Joi.object().keys({
  fullname: Joi.string()
    .min(3)
    .max(30),
  phoneNumber: Joi.string().min(8).max(10).pattern(/^[0-9]+$/),
  email: Joi.string().email({ minDomainSegments: 2 }),
});

const refreshTokenBodyValidation = Joi.object().keys({
  refreshToken: Joi.string().required()
});

exports.Signup = async (req, res) => {
  try {
    const result = userSchema.validate(req.body);
    if (result.error) {
      return responseServerError({ res, err: result.error.message })
    }

    //Check if the username has been already registered.
    var user = await User.findOne({
      username: result.value.username,
    });

    if (user) {
      return responseInValid({ res, message: "Username đã được sử dụng!" });
    }

    const hash = await User.hashPassword(result.value.password);

    const id = uuid(); //Generate unique id for the user.
    result.value.userId = id;

    delete result.value.confirmPassword;
    result.value.password = hash;
    result.value.genToken = uuid();

    const newUser = new User(result.value);
    await newUser.save();

    return responseSuccessWithData({ res, data: "Đăng ký thành công!" });
  } catch (error) {
    return responseServerError({ res, err: "Đăng ký thất bại!" })
  }
};

exports.Activate = async (req, res) => {
  try {
    const { username } = req.body;
    if (!username) {
      return responseServerError({ res, err: "Tài khoản chưa được kích hoạt!" });
    }
    const user = await User.findOne({
      username: username
    });

    if (!user) {
      return responseInValid({ res, message: "Thông tin người dùng không hợp lệ" });
    } else {
      if (user.active) {
        return responseSuccessWithData({ res, data: "Tài khoản đã được kích hoạt" });
      }
      user.active = true;
      await user.save();
      return responseSuccess({ res });
    }
  } catch (error) {
    return responseServerError({ res, err: error.message });
  }
};

exports.Login = async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return responseServerError({ res, err: "Không thể xác thực người dùng." });
    }

    //1. Find if any account with that username exists in DB
    const user = await User.findOne({ username: username });

    // NOT FOUND - Throw error
    if (!user) {
      return responseInValid({ res, message: "Tài khoản không tồn tại" });
    }

    //2. Throw error if account is not activated
    if (!user.active) {
      return responseInValid({ res, message: "Bạn cần kích hoạt tài khoản" });
    }

    //3. Verify the password is valid
    const isValid = await User.comparePasswords(password, user.password);

    if (!isValid) {
      return responseInValid({ res, message: "Sai mật khẩu" });
    }

    //Generate Access token

    const { error, token, refreshToken } = await generateJwt({ username: user.username, id: user.userId, genToken: user.genToken });
    if (error) {
      return responseServerError({ res, err: error.message });
    };
    await user.save();

    //Success
    return responseSuccessWithData({
      res, data: {
        accessToken: token,
        refreshToken: refreshToken,
        user: user
      }
    });
  } catch (err) {
    return responseServerError({ res, err: err.message });
  }
};

exports.RefreshToken = async (req, res) => {
  try {
    const result = refreshTokenBodyValidation.validate(req.body);
    if (result.error) {
      return responseServerError({ res, err: result.error.message })
    }
    const { id } = req.decoded;
    const user = await User.findOne({ userId: id });
    if (!user) {
      return responseServerError({ res, err: "Tài khoản không tồn tại" });
    }
    const { refreshToken } = req.body;
    verifyRefreshToken(refreshToken).then(async ({ tokenDetails }) => {
      const payload = { username: tokenDetails.username, id: tokenDetails.id, genToken: tokenDetails.genToken }
      const response = await generateOnlyJwt(payload);
      if (response.status) {
        return responseSuccessWithData({
          res, data: {
            accessToken: response.token,
            refreshToken: refreshToken
          }
        });
      }
      else {
        return responseServerError({ res, err: response.message });
      }
    }).catch((err) => {
      return responseServerError({ res, err: err.message })
    });
  } catch (err) {
    return responseServerError({ res, err: err.message });
  }
};

exports.ResetPassword = async (req, res) => {
  try {
    const { token, newPassword, confirmPassword } = req.body;
    if (!token || !newPassword || !confirmPassword) {
      return responseInValid({ res, message: "Không thể thực hiện yêu cầu. Vui lòng điền hết các thông tin" });
    }
    const user = await User.findOne({
      resetPasswordToken: req.body.token,
      resetPasswordExpires: { $gt: Date.now() },
    });
    if (!user) {
      return responseInValid({ res, message: "Password reset token is invalid or has expired." });
    }
    if (newPassword !== confirmPassword) {
      return responseInValid({ res, message: "Mật khẩu mới không khớp" });
    }
    const hash = await User.hashPassword(req.body.newPassword);
    user.password = hash;
    user.resetPasswordToken = null;
    user.resetPasswordExpires = "";

    await user.save();
    return responseSuccess({ res });
  } catch (error) {
    return responseServerError({ res, err: error.message });
  }
};

exports.ChangePassword = async (req, res) => {
  try {
    const { id } = req.decoded;
    const { oldPassword, newPassword, confirmPassword } = req.body;
    if (!oldPassword || !newPassword || !confirmPassword) {
      return responseInValid({ res, message: "Không thể thực hiện yêu cầu. Vui lòng điền hết các thông tin" });
    }
    const user = await User.findOne({
      userId: id
    });
    if (!user) {
      return responseInValid({ res, message: "Người dùng không tồn tại." });
    }

    const isValid = await User.comparePasswords(oldPassword, user.password);

    if (!isValid) {
      return responseInValid({ res, message: "Mật khẩu không chính xác" });
    }

    if (oldPassword === newPassword) {
      return responseInValid({ res, message: "Mật khẩu mới cần khác mật khẩu cũ" });
    }

    if (newPassword !== confirmPassword) {
      return responseInValid({ res, message: "Mật khẩu mới không khớp" });
    }
    const hash = await User.hashPassword(newPassword);
    user.password = hash;

    await user.save();

    return responseSuccess({ res });
  } catch (error) {
    return responseServerError({ res, err: error.message });
  }
};

exports.Logout = async (req, res) => {
  try {
    const { id } = req.decoded;

    // let user = await User.findOne({ userId: id });
    // await user.save();

    return responseSuccess({ res });
  } catch (error) {
    return responseServerError({ res, err: error.message });
  }
};

exports.GetUserDetail = async (req, res) => {
  try {
    const { id } = req.decoded;

    let user = await User.findOne({ userId: id });
    return responseSuccessWithData({
      res, data: {
        username: user.username,
        email: user.email,
        fullname: user.fullname,
        phoneNumber: user.phoneNumber,
      }
    });
  } catch (error) {
    return responseServerError({ res, err: error.message });
  }
};

exports.EditUser = async (req, res) => {
  try {
    const { id } = req.decoded;
    const result = EditUserSchema.validate(req.body);
    if (result.error) {
      return responseServerError({ res, err: result.error.message })
    }

    let user = await User.findOneAndUpdate({ userId: id }, req.body, {
      new: true
    }); // return data updated
    return responseSuccessWithData({
      res, data: {
        fullname: user.fullname,
        phoneNumber: user.phoneNumber,
        email: user.email,
        userId: user.userId
      }
    });
  } catch (error) {
    return responseServerError({ res, err: error.message });
  }
};
