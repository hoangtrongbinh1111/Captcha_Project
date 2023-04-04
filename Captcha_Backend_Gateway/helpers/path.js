require('dotenv').config();
const {
    HTTP_HOST,
    HTTP_PORT
} = process.env;

exports.pathPublicImage = ({ folder, imageName }) => {
    return `${HTTP_HOST}:${HTTP_PORT}/images/${folder}/${imageName}`;
}


exports.pathPublicVideo = ({ videoName }) => {
    return `${HTTP_HOST}:${HTTP_PORT}/videos/${videoName}`;
}