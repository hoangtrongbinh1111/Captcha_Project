const express = require("express");
const router = express.Router();
const cleanBody = require("../middlewares/cleanbody");
const datasetController = require("../src/dataset/Dataset.controller");

router.get("/",cleanBody, datasetController.listDataset);
router.post("/create", cleanBody, datasetController.createDataset);
router.get("/read",cleanBody, datasetController.readDataset);
router.patch("/update",cleanBody, datasetController.updateDataset);
router.delete("/delete",cleanBody, datasetController.deleteDataset);
router.get("/review", datasetController.reviewDataset);
module.exports = router;
