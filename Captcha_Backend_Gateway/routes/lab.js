const express = require("express");
const router = express.Router();
const cleanBody = require("../middlewares/cleanbody");
const labController = require("../src/lab/lab.controller");

router.get("/", cleanBody, labController.listLab);
router.post("/create", cleanBody, labController.createLab);
router.get("/read", cleanBody, labController.readLab);
router.delete("/delete", cleanBody, labController.deleteLab);
//api user cấu hình 
router.get("/config",cleanBody,labController.getConfig);
router.patch("/edit",cleanBody,labController.editModelDataAndConfig)
module.exports = router;
