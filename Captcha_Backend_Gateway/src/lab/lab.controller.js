const Joi = require("joi");
require("dotenv").config();
const { v4: uuid } = require("uuid");
const Lab = require("./lab.model");
const Log = require("../log/log.model");
const Model = require("../model/model.model");
const Dataset = require("../dataset/Dataset.model");
const {
    responseServerError,
    responseInValid,
    responseSuccess,
    responseSuccessWithData,
} = require("../../helpers/ResponseRequest");
const path = require("path");
const {
    LAB_FOLDER,
    LAB_SUBFOLDER,
    CONFIG_FOLDER,
    FILE_CFG,
} = require("../../helpers/constant");
const { getDir, removeDir, createFile } = require("../../helpers/file");
const { type } = require("os");
const { config } = require("dotenv");
const { isValidObjectId, Types } = require("mongoose");

const labCreateSchema = Joi.object().keys({
    labName: Joi.string().required(),
    userCreated: Joi.string().required(),
});

const labUpdateSchema = Joi.object().keys({
    labId: Joi.string().required(),
    labName: Joi.string().optional(),
});

const configUpdateSchema = Joi.object().keys({
    labId: Joi.string().required(),
    labName: Joi.string(),
    config: {
        learning_rate: Joi.number(),
        epochs: Joi.number(),
        batch_size: Joi.number(),
        val_size: Joi.number(),
        modelId: Joi.string().required(),
        datasetId: Joi.string().required()
    }
});
exports.listLab = async(req, res) => {
    try {
        let { search, page, limit, from_time, to_time, userCreated } = req.query;
        let options = {};
        if (search && search !== "") {
            options = {
                ...options,
                $or: [
                    { url: new RegExp(search.toString(), "i") },
                    { type: new RegExp(search.toString(), "i") },
                ],
            };
        }
        if (from_time && to_time) {
            options = {
                ...options,
                create_At: {
                    $gte: new Date(from_time).toISOString(),
                    $lt: new Date(to_time).toISOString(),
                },
            };
        }
        if (userCreated) {
            options = {
                ...options,
                userCreated: new RegExp(userCreated.toString(), "i"),
            };
        }

        page = parseInt(page) || 1;
        limit = parseInt(limit) || 10;
        const data = await Lab.find(options).populate('model').populate('dataset')
            .skip((page - 1) * limit)
            .limit(limit)
            .lean()
            .exec();
        const total = await Lab.find(options).countDocuments();
        return responseSuccessWithData({
            res,
            data: {
                data,
                total,
                page,
                last_page: Math.ceil(total / limit),
            },
        });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.createLab = async(req, res) => {
    try {
        const result = labCreateSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message });
        }
        const { labName, userCreated } = req.body;
        const labId = uuid();
        // create folder
        const root = path.resolve("./");
        const labDir = getDir({ dir: root + `/${LAB_FOLDER}/${labId}` });

        const subLabDir = {};
        Object.keys(LAB_SUBFOLDER).map((subfolder) => {
            const dirPath = `/${LAB_FOLDER}/${labId}/${LAB_SUBFOLDER[subfolder]}`;
            getDir({ dir: root + dirPath });
            subLabDir[subfolder] = dirPath;
        });
        // done create

        // tao thang log 
        const logId = uuid();
        const logData = {
            logId
        };
        const newLog = new Log(logData);
        await newLog.save();

        // Thêm vào lab entity
        const labData = {
            labId,
            labName,
            userCreated,
            trainLogPath: subLabDir["trainLogPath"],
            testLogPath: subLabDir["testLogPath"],
            trainedModelPath: subLabDir["trainedModelPath"],
            logId: logId
        };
        const newLab = new Lab(labData);
        await newLab.save();

        return responseSuccessWithData({ res, data: newLab });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.readLab = async(req, res) => {
    try {
        const { labId } = req.query;
        let labItem = await Lab.findOne({ labId: labId }).populate('model').populate('dataset');
        if (labItem) {
            return responseSuccessWithData({ res, data: labItem });
        } else {
            return responseServerError({ res, err: "Lab not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.deleteLab = async(req, res) => {
    try {
        const { labId } = req.query;

        var labItem = await Lab.findOne({
            labId: labId,
        });
        if (!labItem) {
            return responseServerError({ res, err: "Lab không tồn tại!" });
        }
        await Lab.deleteOne({ labId: labId });
        // delete folder
        const root = path.resolve("./");
        const labDir = removeDir({
            dir: root + `/${LAB_FOLDER}/${labId}`,
        });
        // done delete
        return responseSuccess({ res });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};

exports.getConfig = async(req, res) => {
    try {
        const { labId } = req.query;
        var labItem = await Lab.findOne({ labId: labId }, "config");
        if (labItem) {
            return responseSuccessWithData({ res, data: labItem });
        } else {
            return responseServerError({ res, err: "Lab not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.editModelDataAndConfig = async(req, res) => {
    try {
        const result = configUpdateSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message });
        }
        const { labId, config, labName } = req.body;

        var labItem = await Lab.findOne({ labId: labId });
        if (!labItem) {
            return responseServerError({ res, err: "Lab not found" });
        }
        delete result.value.labId;
        let configUpdate = null;
        const modelData = await Model.findOne({ modelId: config.modelId });
        const datasetData = await Dataset.findOne({ datasetId: config.datasetId });
        if (labName) {
            configUpdate = await Lab.findOneAndUpdate({ labId: labId }, { $set: { config: config, labName: labName,  model: Types.ObjectId(modelData._id), dataset: Types.ObjectId(datasetData._id) } }, {
                new: true,
            });
        } else {
            configUpdate = await Lab.findOneAndUpdate({ labId: labId }, { $set: { config: config, model: Types.ObjectId(modelData._id), dataset: Types.ObjectId(datasetData._id) } }, {
                new: true,
            });
        }
        
        return responseSuccessWithData({
            res,    
            data: {
                configUpdate,
        }});
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};

