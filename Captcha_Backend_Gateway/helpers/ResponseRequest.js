exports.responseSuccess = ({ res }) => {
    return res.json({
        status: true,
        message: "Success"
    });
}

exports.responseServerError = ({ res, err }) => {
    return res.status(500).json({
        status: false,
        error: err
    });
}

exports.responseInValid = ({ res, message }) => {
    return res.status(422).json({
        status: 422,
        message: message
    });
}

exports.responseSuccessWithData = ({ res, data }) => {
    return res.json({
        status: true,
        data: data,
        message: "Success"
    });
}