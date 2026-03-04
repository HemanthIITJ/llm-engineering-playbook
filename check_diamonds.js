const fs = require('fs');
const content = fs.readFileSync('src/generated/content-manifest.json', 'utf8');
if (content.includes('\uFFFD')) {
    console.log("FOUND U+FFFD (diamond question marks) in the generated content!");
} else {
    console.log("Clean! No diamond question marks found.");
}
