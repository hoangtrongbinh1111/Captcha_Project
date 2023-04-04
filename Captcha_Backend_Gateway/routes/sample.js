const express = require("express");
const router = express.Router();
const cleanBody = require("../middlewares/cleanbody");
const sampleController = require("../src/sample/Sample.controller");

router.get("/",cleanBody, sampleController.listsample);
router.get("/read",cleanBody, sampleController.readsample);
router.patch("/update",cleanBody, sampleController.updatesample);
router.delete("/delete",cleanBody, sampleController.deletesample);

module.exports = router;
