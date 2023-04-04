const mongoose = require("mongoose");
const Schema = mongoose.Schema;
const {random} = require("../../helpers/rand");

const labSchema = new Schema(
    {
        labId: { type: String, unique: true, required: true },
        labName: { type: String, required: true},
        userCreated: { type: String, default: null},
        config:  {
            learning_rate: { type: Number, default: 0.001 },
            epochs: { type: Number, default: 2 },
            batch_size: { type: Number, default: 8 },
            val_size: { type: Number, default: 0.2 },
            modelId: { type: Schema.Types.String },
            datasetId: { type: Schema.Types.String }
        },
        logId: {type:String,default: null},
        model: { type: Schema.Types.ObjectId, ref: 'model' },
        dataset: { type: Schema.Types.ObjectId, ref: 'dataset' }
    },
    {
        timestamps: {
            createdAt: "createdAt",
            updatedAt: "updatedAt",
        },
    }
);

const Lab = mongoose.model("lab", labSchema);
module.exports = Lab;