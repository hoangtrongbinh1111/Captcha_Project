const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const Filter = new Schema({
    filterId: { type: String, required: true, unique: true },
    filterName: { type: String, required: true },
    userUpload: { type: String, required: true },
    savePath: { type: String, default: null },
    desc: { type: String, default: null },
}, {
    timestamps: {
        createdAt: "createdAt",
        updatedAt: "updatedAt",
    }
})
const filter = mongoose.model('filter', Filter)
module.exports = filter