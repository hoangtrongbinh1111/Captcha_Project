const Joi = require("joi"); //validate
require("dotenv").config();
const { v4: uuid } = require("uuid"); //gen id
const Filter = require("./Filter.model");
const csv = require('csvtojson');
const {
    responseServerError,
    responseSuccess,
    responseInValid,
    responseSuccessWithData,
} = require("../../helpers/ResponseRequest"); //response server
const path = require("path"); //work with path
const { getDir, removeDir } = require("../../helpers/file"); // create dir
const { FILTER_DATA_FOLDER, DATA_SUBFOLDER } = require("../../helpers/constant");

const updatefilterSchema = Joi.object().keys({
    filterId: Joi.string().required(),
    filterName: Joi.string().optional(),
    desc: Joi.string().optional(),
});
const filterCreateSchema = Joi.object().keys({
    userUpload: Joi.string().required(),
    filterName: Joi.string().required(),
});
exports.listfilter = async (req, res) => {
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
        const data = await Filter
            .find(options)
            .skip((page - 1) * limit)
            .limit(limit)
            .lean()
            .exec();
        const total = await Filter.find(options).countDocuments();
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

exports.updatefilter = async (req, res) => {
    try {
        const result = updatefilterSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message });
        }
        const { filterId, filterName, desc } = req.body;

        var filterItem = await Filter.findOne({ filterId: filterId });
        if (!filterItem) {
            return responseServerError({ res, err: "filter not found" });
        }
        delete result.value.filterId;
        let filterlUpdate = await Filter.findOneAndUpdate({ filterId: filterId },
            result.value, {
            new: true,
        }
        );
        return responseSuccessWithData({
            res,
            data: filterlUpdate,
        });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};

exports.readfilter = async (req, res) => {
    try {
        const { filterId } = req.query;
        let filterItem = await Filter.findOne({ filterId: filterId });
        if (filterItem) {
            return responseSuccessWithData({ res, data: filterItem });
        } else {
            return responseServerError({ res, err: "filter not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.deletefilter = async (req, res) => {
    try {
        const { filterId } = req.query;

        var filterItem = await Filter.findOne({
            filterId: filterId,
        });
        if (!filterItem) {
            return responseServerError({ res, err: "filter không tồn tại!" });
        }
        await Filter.deleteOne({ filterId: filterId });
        //delete folder
        const root = path.resolve("./");
        const dataDir = removeDir({
            dir: root + `/${FILTER_DATA_FOLDER}/${filterId}`,
        });
        //end delete folder
        return responseSuccess({ res });
    } catch (err) {
        return responseServerError({ res, err: err.message });
    }
};