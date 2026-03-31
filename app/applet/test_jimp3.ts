import { Jimp } from "jimp";

async function test() {
    const img = new Jimp({ width: 10, height: 10 });
    try {
        const b64 = await img.getBase64("image/png");
        console.log("getBase64 worked:", b64.substring(0, 20));
    } catch (e) {
        console.error("getBase64 failed:", e);
    }
}
test();
