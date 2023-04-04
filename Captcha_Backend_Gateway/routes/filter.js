const express = require("express");
const router = express.Router();
const cleanBody = require("../middlewares/cleanbody");
const filterController = require("../src/filter/Filter.controller");

router.get("/", cleanBody, filterController.listfilter);
router.get("/read", cleanBody, filterController.readfilter);
router.patch("/update", cleanBody, filterController.updatefilter);
router.delete("/delete", cleanBody, filterController.deletefilter);

module.exports = router;
