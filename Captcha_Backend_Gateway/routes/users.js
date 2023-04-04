const express = require("express");
const router = express.Router();

const cleanBody = require("../middlewares/cleanbody");
const { validateToken } = require("../middlewares/validateToken");

const AuthController = require("../src/users/user.controller");

router.post("/signup", cleanBody, AuthController.Signup);

router.post("/refresh", validateToken, AuthController.RefreshToken);

router.patch("/activate", cleanBody, AuthController.Activate);

router.post("/login", cleanBody, AuthController.Login);

router.patch("/reset", cleanBody, AuthController.ResetPassword);

router.patch("/change", validateToken, cleanBody, AuthController.ChangePassword);

router.patch("/edit", validateToken, cleanBody, AuthController.EditUser);

router.get("/logout", validateToken, AuthController.Logout);

router.get("/detail", validateToken, AuthController.GetUserDetail);

module.exports = router;
