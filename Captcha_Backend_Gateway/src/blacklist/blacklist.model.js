const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const blacklistSchema = new Schema(
  {
    blacklistId: { type: String, unique: true, required: true },
    url: {type: String, unique: true, required: true},
    type: {type: String, required: false}
  },
  {
    timestamps: {
      createdAt: "createdAt",
      updatedAt: "updatedAt",
    },
  }
);

const BlackList = mongoose.model("blacklist", blacklistSchema);
module.exports = BlackList;
