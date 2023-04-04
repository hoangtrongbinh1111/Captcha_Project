const fs = require('fs')
const path = require('path')

exports.getDir = ({ dir }) => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    return dir;
}

exports.removeDir = ({ dir }) => {
    fs.rmSync(dir, { recursive: true, force: true });
}

exports.createFile = ({ dir }) => {
    fs.open(dir, 'w', (err) => {
        if (err) throw err;
    });
}

exports.saveFileInfer = ({ filename, file }) => {
    fs.writeFileSync(filename, file, { encoding: 'base64' }, (err) => {
        if (err) throw err;
    });
}

