const Joi = require("joi"); //validate
require("dotenv").config();
const { v4: uuid } = require("uuid"); //gen id
const dataset = require("./Dataset.model");
const csv=require('csvtojson');
const {
    responseServerError,
    responseSuccess,
    responseInValid,
    responseSuccessWithData,
} = require("../../helpers/ResponseRequest"); //response server
const path = require("path"); //work with path
const { getDir, removeDir } = require("../../helpers/file"); // create dir
const { DATA_FOLDER, DATA_SUBFOLDER } = require("../../helpers/constant");
const { VirtualType } = require("mongoose");
const dataSet = require("./Dataset.model");

const updateDatasetSchema = Joi.object().keys({
    datasetId: Joi.string().required(),
    numTrain: Joi.number().optional(),
    numVal: Joi.number().optional(),
    numTest: Joi.number().optional(),
    dataType: Joi.string().optional(),
    desc: Joi.string().optional(),
});
const datasetCreateSchema = Joi.object().keys({
    userUpload: Joi.string().required(),
    dataName: Joi.string().required(),
});
exports.listDataset = async (req, res) => {
    try {
        let { search, page, limit, from_time, to_time } = req.query;
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

        page = parseInt(page) || 1;
        limit = parseInt(limit) || 10;
        const data = await dataset
            .find(options)
            .skip((page - 1) * limit)
            .limit(limit)
            .lean()
            .exec();
        const total = await dataset.find(options).countDocuments();
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

exports.reviewDataset = async (req, res) => {
    try {
        let { datasetId, maxSample } = req.query;
        const datasetData = await dataset.findOne({ datasetId: datasetId });
        const csvFilePath= `${datasetData.savePath}/train.csv`;
        csv()
            .fromFile(csvFilePath)
            .then((samplePreview)=>{
                const listDataSample = samplePreview.slice(0, maxSample ?? 10);
                return responseSuccessWithData({
                    res,
                    data: {
                        listDataSample
                    },
                });
            })
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.createDataset = async (req, res) => {
    try {
        const result = datasetCreateSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message });
        }
        const { dataName, userUpload } = req.body;
        const datasetId = uuid();
        
        const savePath = `/${DATA_FOLDER}/${datasetId}`;
        //create folder
        const root = path.resolve("./");
        // const dir = getDir({ dir: root + `/${DATA_FOLDER}` });
        const dataDir = getDir({
            dir: root + `/${DATA_FOLDER}/${datasetId}`,
        });
        Object.keys(DATA_SUBFOLDER).map((subfolder) => {
            getDir({
                dir: root + `/${DATA_FOLDER}/${datasetId}/${DATA_SUBFOLDER[subfolder]}`,
            });
        });
        //end create folder
        const data = {
            datasetId,
            dataName,
            userUpload,
            savePath,
        };
        const newData = new dataset(data);
        await newData.save();
        return responseSuccessWithData({ res, data: newData });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};
exports.updateDataset = async (req, res) => {
    try {
        const result = updateDatasetSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message });
        }
        const { datasetId, numTrain, numVal, numTest, dataType, desc } = req.body;

        var datasetItem = await dataset.findOne({ datasetId: datasetId });
        if (!datasetItem) {
            return responseServerError({ res, err: "Dataset not found" });
        }
        delete result.value.datasetId;
        let datasetlUpdate = await dataset.findOneAndUpdate({ datasetId: datasetId },
            result.value, {
            new: true,
        }
        );
        return responseSuccessWithData({
            res,
            data: datasetlUpdate,
        });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};

exports.readDataset = async (req, res) => {
    try {
        const { datasetId } = req.query;
        let datasetItem = await dataset.findOne({ datasetId: datasetId });
        if (datasetItem) {
            return responseSuccessWithData({ res, data: datasetItem });
        } else {
            return responseServerError({ res, err: "Dataset not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.deleteDataset = async (req, res) => {
    try {
        const { datasetId } = req.query;

        var datasetItem = await dataset.findOne({
            datasetId: datasetId,
        });
        if (!datasetItem) {
            return responseServerError({ res, err: "Dataset không tồn tại!" });
        }
        await dataset.deleteOne({ datasetId: datasetId });
        //delete folder
        const root = path.resolve("./");
        const dataDir = removeDir({
            dir: root + `/${DATA_FOLDER}/${datasetId}`,
        });
        //end delete folder
        return responseSuccess({ res });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};

exports.datasetCreateSchema