const express = require("express");
const router = express.Router();
const { validateToken } = require("../middlewares/validateToken");
const cleanBody = require("../middlewares/cleanbody");
const ModelController = require("../src/model/model.controller");

// router.get("/", validateToken, ModelController.ListModels);
// router.get("/detail", validateToken, ModelController.DetailModel);
// router.post("/add", validateToken, cleanBody, ModelController.AddModels);
// router.patch("/edit", validateToken, cleanBody, ModelController.EditModel);
// router.delete("/delete", validateToken, ModelController.DeleteModel);
router.get("/", cleanBody, ModelController.listModel);
router.post("/create", cleanBody, ModelController.createModel);
router.get("/read",cleanBody, ModelController.readModel);
router.patch("/update",cleanBody, ModelController.updateModel);
router.delete("/delete",cleanBody, ModelController.deleteModel);

module.exports = router;
