const Joi = require("joi");
const { v4: uuid } = require("uuid");
const Model = require("../model/model.model");
const Users = require("../users/user.model");
const { checkAuthorize } = require("../../middlewares/checkAuthorize");
const {
    responseServerError,
    responseInValid,
    responseSuccess,
    responseSuccessWithData,
} = require("../../helpers/ResponseRequest");

const createModelSchema = Joi.object().keys({
  modelName: Joi.string().required(),
  userCreated: Joi.string(),
});
const updateModelSchema = Joi.object().keys({
  modelId: Joi.string().required(),
  modelName: Joi.string(),
  userCreated: Joi.string(),
});
exports.listModel = async (req, res) => {
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
        const data = await Model.find(options)
            .skip((page - 1) * limit)
            .limit(limit)
            .lean()
            .exec();
        const total = await Model.find(options).countDocuments();
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
exports.createModel = async (req, res) => {
  try {
    const result = createModelSchema.validate(req.body);
    if (result.error) {
      return responseServerError({ res, err: result.error.message });
    }
    const { modelName, userCreated } = req.body;
    const modelId = uuid();

    const modelData = {
      modelId,
      modelName,
      userCreated,
    };
    const newModel = new Model(modelData);
    await newModel.save();
    return responseSuccessWithData({ res, data: newModel });
  } catch (err) {
    return responseServerError({ res, err: err.message });
  }
};
exports.updateModel = async (req, res) => {
  try {
    const result = updateModelSchema.validate(req.body);
    if (result.error) {
      return responseServerError({ res, err: result.error.message });
    }
    const { modelId, modelName, userCreated } = req.body;
    var modelItem = Model.findOne({ modelId: modelId });
    if (!modelItem) {
      return responseServerError({ res, err: "Model not found" });
    }
}catch (error) {
  return responseServerError({ res, err: error.message });
}
}

exports.readModel = async (req, res) => {
    try {
        const { modelId } = req.query;
        let modelItem = await Model.findOne({ modelId: modelId });
        if (modelItem) {
            return responseSuccessWithData({ res, data: modelItem });
        } else {
            return responseServerError({ res, err: "Model not found" });
        }
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

exports.deleteModel = async (req, res) => {
    try {
        const { modelId } = req.query;
        //Check if the username has been already registered.
        var modelItem = await Model.findOne({
            modelId: modelId,
        });

        if (!modelItem) {
            return responseServerError({ res, err: "Model không tồn tại!" });
        }

        await Model.deleteOne({ modelId: modelId });
        return responseSuccess({ res });
    } catch (error) {
        return responseServerError({ res, err: error.message });
    }
};

// exports.ListModels = async (req, res) => {
//     try {
//         const { id } = req.decoded;
//         checkAuthorize(id, res);
//         let { search, page, limit } = req.query;
//         let options = {
//         };
//         if (search && search !== "") {
//             options = {
//                 ...options,
//                 $or: [
//                     { modelId: new RegExp(search.toString(), 'i') },
//                     { urlSaveModel: new RegExp(search.toString(), 'i') },
//                     { params: new RegExp(search.toString(), 'i') },
//                     { desc: new RegExp(search.toString(), 'i') },
//                 ]
//             };
//         }

//         page = parseInt(page) || 1;
//         limit = parseInt(limit) || 10;
//         const data = await Model.find(options).skip((page - 1) * limit).limit(limit).lean().exec();
//         const total = await Model.find(options).countDocuments();
//         return responseSuccessWithData({
//             res, data: {
//                 data,
//                 total,
//                 page,
//                 last_page: Math.ceil(total / limit)
//             }
//         });
//     } catch (error) {
//         return responseServerError({ res, err: error.message });
//     }
// };

// exports.DetailModel = async (req, res) => {
//     try {
//         const { id } = req.decoded;
//         checkAuthorize(id, res);
//         const { modelId } = req.query;
//         let modelItem = await Model.findOne({ modelId: modelId });
//         if (modelItem) {
//             return responseSuccessWithData({ res, data: modelItem });
//         }
//         return responseServerError({ res, err: "Model không tồn tại!" });
//     } catch (error) {
//         return responseServerError({ res, err: error.message });
//     }
// };

// exports.AddModels = async (req, res) => {
//     try {
//         const { id } = req.decoded;
//         checkAuthorize(id, res);

//         const result = AddModelSchema.validate(req.body);
//         if (result.error) {
//             return responseServerError({ res, err: result.error.message })
//         }

//         //Check if the username has been already registered.
//         var modelItem = await Model.findOne({
//             urlSaveModel: result.value.urlSaveModel,
//         });

//         if (modelItem) {
//             return res.status(400).json({
//                 status: false,
//                 message: "Model đã có trong hệ thống!",
//             });
//         }

//         const modelId = uuid();
//         result.value.modelId = modelId;

//         const newModel = new Model(result.value);
//         await newModel.save();

//         return res.status(200).json({
//             status: true,
//             message: "Thêm model thành công!"
//         });
//     } catch (error) {
//         return responseServerError({ res, err: error.message });
//     }
// };

// exports.EditModel = async (req, res) => {
//     try {
//         const { id } = req.decoded;
//         checkAuthorize(id, res);
//         const result = EditModelSchema.validate(req.body);
//         if (result.error) {
//             return responseServerError({ res, err: result.error.message })
//         }
//         const { modelId, urlSaveModel, params, desc } = req.body;
//         //Check if the username has been already registered.
//         var modelItem = await Model.findOne({
//             modelId: modelId,
//         });

//         if (!modelItem) {
//             return res.status(400).json({
//                 status: false,
//                 message: "Model không tồn tại!",
//             });
//         }
//         delete result.value.modelId
//         let modelUpdate = await Model.findOneAndUpdate({ modelId: modelId }, result.value, {
//             new: true
//         }); // return data updated

//         return res.status(200).send({ status: true, data: modelUpdate });
//     } catch (error) {
//         return responseServerError({ res, err: error.message });
//     }
// };

// exports.DeleteModel = async (req, res) => {
//     try {
//         const { id } = req.decoded;
//         checkAuthorize(id, res);

//         const { modelId } = req.query;
//         //Check if the username has been already registered.
//         var modelItem = await Model.findOne({
//             modelId: modelId,
//         });

//         if (!modelItem) {
//             return res.status(400).json({
//                 status: false,
//                 message: "modelId không tồn tại!",
//             });
//         }

//         await Model.deleteOne({ modelId: modelId }); // return data updated
//         return res.status(200).send({ status: true, data: null });
//     } catch (error) {
//         return responseServerError({ res, err: error.message });
//     }
// };