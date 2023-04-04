const Joi = require("joi");
require("dotenv").config();
const { v4: uuid } = require("uuid");
const { responseServerError, responseInValid, responseSuccessWithData } = require("../../helpers/ResponseRequest");
const { MALICIOUS, UNMALICIOUS } = require("../../helpers/constant");
const { checkAuthorize } = require("../../middlewares/checkAuthorize");
const Users = require("../users/user.model");
const BlackList = require("../blacklist/blacklist.model");
const Log = require("./log.model");
const Lab = require("../lab/lab.model");

const logDetectSchema = Joi.object().keys({
    url: Joi.string().required()
});

exports.ListLogs = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);
        let { search, page, limit, from_time, to_time, userId } = req.query;
        let options = {};
        if (userId && search !== "") {
            options = {
                ...options,
                userId: userId
            }
        }
        if (search && search !== "") {
            options = {
                ...options,
                $or: [
                    { url: new RegExp(search.toString(), 'i') },
                    { result: new RegExp(search.toString(), 'i') },
                    { userId: new RegExp(search.toString(), 'i') },
                ]
            };
        }
        if (from_time && to_time) {
            options = {
                ...options,
                timeExecute: {
                    $gte: new Date(from_time).toISOString(),
                    $lt: new Date(to_time).toISOString()
                }
            }
        }

        page = parseInt(page) || 1;
        limit = parseInt(limit) || 10;
        const data = await Log.find(options).skip((page - 1) * limit).limit(limit).lean().exec();
        const total = await Log.find(options).countDocuments();
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


exports.readLog = async(req, res) => {
    try {
        const { labId } = req.query;
        let {config, logId} = await Lab.findOne({ labId: labId });
        let logData = await Log.findOne({ logId: logId });
        if (logData) {
            return responseSuccessWithData({ res, data: {
                logData: logData,
                trainConfig: config
            } });
        } else {
            return responseServerError({ res, err: "Lab not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.DetectMaliciousWebsite = async (req, res) => {
    try {
        const { id } = req.decoded;
        checkAuthorize(id, res);

        const result = logDetectSchema.validate(req.body);
        if (result.error) {
            return responseServerError({ res, err: result.error.message })
        }

        const { url } = req.body;
        const blList = await BlackList.findOne({
            url: url,
        });

        if (blList) {
            return responseSuccessWithData({ res, data: MALICIOUS }); // đường dẫn độc hại
        }

        // ** HANDLE DETECT URL MALICIOUS */
        const resultDetect = MALICIOUS;
        // ** END HANDLE DETECT */

        // Thêm vào logs
        const logsData = {
            logId: uuid(),
            url: url,
            timeExecute: "10",
            result: resultDetect,
            userId: id
        }
        const newLog = new Log(logsData);
        await newLog.save();

        return responseSuccessWithData({ res, data: resultDetect });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};