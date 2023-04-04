const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const sample = new Schema({
    sampleId: { type: String, required: true, unique: true },
    sampleName: {type: String, required: true},
    userUpload: { type: String, required: true },
    savePath: { type: String, default: null },
    desc: { type: String, default: null },
}, {
    timestamps: {
        createdAt: "createdAt",
        updatedAt: "updatedAt",
    }
})
const Sample = mongoose.model('sample', sample)
module.exports = Sample