import { Jimp } from 'jimp';
import fs from 'fs';

async function createImages() {
    const cover = new Jimp({ width: 64, height: 64, color: 0xFF0000FF });
    const watermark = new Jimp({ width: 8, height: 8, color: 0x00FF00FF });
    
    const coverBuf = await cover.getBuffer("image/png");
    const watermarkBuf = await watermark.getBuffer("image/png");
    
    fs.writeFileSync('cover.png', coverBuf);
    fs.writeFileSync('watermark.png', watermarkBuf);
    console.log("Images created");
}
createImages();
