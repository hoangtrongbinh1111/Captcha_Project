var crypto = require('crypto');

exports.hashMD5 = (string) => {
    return crypto.createHash('md5').update(string).digest("hex");
}