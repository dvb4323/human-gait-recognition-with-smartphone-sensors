/**
 * Metro configuration for React Native
 * Updated to support .tflite files
 */

const {getDefaultConfig} = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add .tflite to asset extensions so Metro bundles them
config.resolver.assetExts.push('tflite');

module.exports = config;
