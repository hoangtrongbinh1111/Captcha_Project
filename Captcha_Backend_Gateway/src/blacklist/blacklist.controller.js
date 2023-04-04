const Joi = require("joi");
require("dotenv").config();
const { v4: uuid } = require("uuid");
const { responseServerError, responseInValid, responseSuccessWithData, responseSuccess } = require("../../helpers/ResponseRequest");
const { checkAuthorize } = require("../../middlewares/checkAuthorize");
const Users = require("../users/user.model");
const BlackList = require("./blacklist.model");

const AddBlackListSchema = Joi.object().keys({
    url: Joi.string().required(),
    type: Joi.string()
});

const EditBlackListSchema = Joi.object().keys({
    blacklistId: Joi.string().required(),
    url: Joi.string(),
    type: Joi.string()
});

exports.ListBlackLists = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);
        let { search, page, limit, from_time, to_time } = req.query;
        let options = {};
        if (search && search !== "") {
            options = {
                ...options,
                $or: [
                    { url: new RegExp(search.toString(), 'i') },
                    { type: new RegExp(search.toString(), 'i') },
                ]
            };
        }
        if (from_time && to_time) {
            options = {
                ...options,
                create_At: {
                    $gte: new Date(from_time).toISOString(),
                    $lt: new Date(to_time).toISOString()
                }
            }
        }

        page = parseInt(page) || 1;
        limit = parseInt(limit) || 10;
        const data = await BlackList.find(options).skip((page - 1) * limit).limit(limit).lean().exec();
        const total = await BlackList.find(options).countDocuments();
        return responseSuccessWithData({
            res, data: {
                data,
                total,
                page,
                last_page: Math.ceil(total / limit)
            }
        });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.DetailBlackLists = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);
        const { blacklistId } = req.query;
        let blItem = await BlackList.findOne({ blacklistId: blacklistId });
        if (blItem) {
            return responseSuccessWithData({ res, data: blItem });
        }
        else {
            return responseServerError({ res, err: "Blacklist không tồn tại!" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.AddBlackList = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);
        const result = AddBlackListSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message })
        }

        //Check if the username has been already registered.
        var blList = await BlackList.findOne({
            url: result.value.url,
        });

        if (blList) {
            return responseServerError({ res, err: "Url đã có trong hệ thống!" });
        }

        const blacklistId = uuid(); //Generate unique id for blacklist.
        result.value.blacklistId = blacklistId;

        const newBlList = new BlackList(result.value);
        await newBlList.save();

        return responseSuccess({ res });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.EditBlackList = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);

        const result = EditBlackListSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message })
        }
        const { blacklistId, url, type } = req.body;
        //Check if the username has been already registered.
        var blacklistItem = await BlackList.findOne({
            blacklistId: blacklistId,
        });

        if (!blacklistItem) {
            return responseServerError({ res, err: "blacklistId không tồn tại!" })
        }
        delete result.value.blacklistId
        let blUpdate = await BlackList.findOneAndUpdate({ blacklistId: blacklistId }, result.value, {
            new: true
        }); // return data updated
        return responseSuccessWithData({ res, data: blUpdate });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.DeleteBlackList = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);

        const { blacklistId } = req.query;
        //Check if the username has been already registered.
        var blacklistItem = await BlackList.findOne({
            blacklistId: blacklistId,
        });

        if (!blacklistItem) {
            return responseServerError({ res, err: "blacklistId không tồn tại!" });
        }

        await BlackList.deleteOne({ blacklistId: blacklistId }); // return data updated
        return responseSuccess({ res });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};