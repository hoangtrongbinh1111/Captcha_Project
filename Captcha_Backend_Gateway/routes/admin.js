const express = require("express");
const router = express.Router();
const { validateToken } = require("../middlewares/validateToken");
const cleanBody = require("../middlewares/cleanbody");
const AdminController = require("../src/admin/admin.controller");
const cleanbody = require("../middlewares/cleanbody");




router.get("/user/list", validateToken, AdminController.ListUsers);
router.post("/blockuser", validateToken, cleanBody, AdminController.BlockUser);
module.exports = router;
