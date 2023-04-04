const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const modelSchema = new Schema(
    {
        modelId: { type: String, unique: true, required: true },
        modelName: { type: String, required: true },
        userCreated: { type: String, default: null },
        desc: {type: String, default: null}
    },
    {
        timestamps: {
            createdAt: "createdAt",
            updatedAt: "updatedAt",
        },
    }
);

const Model = mongoose.model("model", modelSchema);
module.exports = Model;
