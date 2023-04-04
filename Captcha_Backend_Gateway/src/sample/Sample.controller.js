const Joi = require("joi"); //validate
require("dotenv").config();
const { v4: uuid } = require("uuid"); //gen id
const Sample = require("./Sample.model");
const csv = require('csvtojson');
const {
    responseServerError,
    responseSuccess,
    responseInValid,
    responseSuccessWithData,
} = require("../../helpers/ResponseRequest"); //response server
const path = require("path"); //work with path
const { getDir, removeDir } = require("../../helpers/file"); // create dir
const { MODEL_SAMPLE_FOLDER, DATA_SUBFOLDER } = require("../../helpers/constant");

const updatesampleSchema = Joi.object().keys({
    sampleId: Joi.string().required(),
    sampleName: Joi.string().optional(),
    desc: Joi.string().optional(),
});
const sampleCreateSchema = Joi.object().keys({
    userUpload: Joi.string().required(),
    sampleName: Joi.string().required(),
});
exports.listsample = async (req, res) => {
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
        const data = await Sample
            .find(options)
            .skip((page - 1) * limit)
            .limit(limit)
            .lean()
            .exec();
        const total = await Sample.find(options).countDocuments();
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

exports.updatesample = async (req, res) => {
    try {
        const result = updatesampleSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message });
        }
        const { sampleId, sampleName, desc } = req.body;

        var sampleItem = await Sample.findOne({ sampleId: sampleId });
        if (!sampleItem) {
            return responseServerError({ res, err: "sample not found" });
        }
        delete result.value.sampleId;
        let samplelUpdate = await Sample.findOneAndUpdate({ sampleId: sampleId },
            result.value, {
            new: true,
        }
        );
        return responseSuccessWithData({
            res,
            data: samplelUpdate,
        });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};

exports.readsample = async (req, res) => {
    try {
        const { sampleId } = req.query;
        let sampleItem = await Sample.findOne({ sampleId: sampleId });
        if (sampleItem) {
            return responseSuccessWithData({ res, data: sampleItem });
        } else {
            return responseServerError({ res, err: "sample not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.deletesample = async (req, res) => {
    try {
        const { sampleId } = req.query;

        var sampleItem = await Sample.findOne({
            sampleId: sampleId,
        });
        if (!sampleItem) {
            return responseServerError({ res, err: "sample không tồn tại!" });
        }
        await Sample.deleteOne({ sampleId: sampleId });
        //delete folder
        const root = path.resolve("./");
        const dataDir = removeDir({
            dir: root + `/${MODEL_SAMPLE_FOLDER}/${sampleId}`,
        });
        //end delete folder
        return responseSuccess({ res });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};