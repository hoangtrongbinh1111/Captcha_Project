/**
 * @file docker-entry for development container
 * @author Redsandro (https://www.windowsremix.com/)
 */
'use strict'
const { spawn } = require('child_process')

/*
 * Install dependencies every time package.json changes
 */
spawn('nodemon -w package.json --exec "npm install"', {
    stdio: 'inherit',
    shell: true
})

/*
 * Restart node when a source file changes, plus:
 * Restart when `npm install` ran based on `package-lock.json` changing.
 */
spawn('nodemon --inspect -e js,json -i node_modules -i package.json app.js', {
    stdio: 'inherit',
    shell: true
})