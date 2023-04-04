const express = require("express");
const router = express.Router();
const { validateToken } = require("../middlewares/validateToken");
const cleanBody = require("../middlewares/cleanbody");
const BlackListController = require("../src/blacklist/blacklist.controller");

router.get("/", validateToken, BlackListController.ListBlackLists);
router.get("/detail", validateToken, BlackListController.DetailBlackLists);
router.post("/add", validateToken, cleanBody, BlackListController.AddBlackList);
router.patch("/edit", validateToken, cleanBody, BlackListController.EditBlackList);
router.delete("/delete", validateToken, BlackListController.DeleteBlackList);

module.exports = router;
