import express from "express";
import path from "path";
import multer from "multer";
import { Jimp } from "jimp";
import { Matrix, SingularValueDecomposition } from "ml-matrix";
import fs from "fs";

const app = express();
const PORT = 3000;

// Configure multer for memory storage
const upload = multer({ storage: multer.memoryStorage() });

function intToRGBA(i: number) {
    return {
        r: (i >> 24) & 0xff,
        g: (i >> 16) & 0xff,
        b: (i >> 8) & 0xff,
        a: i & 0xff,
    };
}

function rgbaToInt(r: number, g: number, b: number, a: number) {
    return (
        ((r & 0xff) << 24) |
        ((g & 0xff) << 16) |
        ((b & 0xff) << 8) |
        (a & 0xff)
    ) >>> 0;
}

function rgbToY(r: number, g: number, b: number) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

function yCbCrToRgb(y: number, cb: number, cr: number) {
    const r = Math.max(0, Math.min(255, Math.round(y + 1.402 * (cr - 128))));
    const g = Math.max(0, Math.min(255, Math.round(y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128))));
    const b = Math.max(0, Math.min(255, Math.round(y + 1.772 * (cb - 128))));
    return { r, g, b };
}

app.use(express.json({ limit: '50mb' }));

async function subbandToBase64(subband: number[][]): Promise<string> {
    const rows = subband.length;
    const cols = subband[0].length;
    const img = new Jimp({ width: cols, height: rows });

    // Find min and max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (subband[i][j] < min) min = subband[i][j];
            if (subband[i][j] > max) max = subband[i][j];
        }
    }

    const range = max - min || 1;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const val = Math.round(((subband[i][j] - min) / range) * 255);
            img.setPixelColor(rgbaToInt(val, val, val, 255), j, i);
        }
    }

    return await img.getBase64("image/png");
}

// --- Watermarking Logic ---

/**
 * 2D Haar Discrete Wavelet Transform
 */
function dwt2(data: number[][]): { LL: number[][], LH: number[][], HL: number[][], HH: number[][] } {
    const rows = data.length;
    const cols = data[0].length;
    const halfRows = Math.floor(rows / 2);
    const halfCols = Math.floor(cols / 2);

    const LL = Array.from({ length: halfRows }, () => new Array(halfCols).fill(0));
    const LH = Array.from({ length: halfRows }, () => new Array(halfCols).fill(0));
    const HL = Array.from({ length: halfRows }, () => new Array(halfCols).fill(0));
    const HH = Array.from({ length: halfRows }, () => new Array(halfCols).fill(0));

    for (let i = 0; i < halfRows; i++) {
        for (let j = 0; j < halfCols; j++) {
            const a = data[2 * i][2 * j];
            const b = data[2 * i][2 * j + 1];
            const c = data[2 * i + 1][2 * j];
            const d = data[2 * i + 1][2 * j + 1];

            LL[i][j] = (a + b + c + d) / 2;
            LH[i][j] = (a - b + c - d) / 2;
            HL[i][j] = (a + b - c - d) / 2;
            HH[i][j] = (a - b - c + d) / 2;
        }
    }

    return { LL, LH, HL, HH };
}

/**
 * Inverse 2D Haar Discrete Wavelet Transform
 */
function idwt2(LL: number[][], LH: number[][], HL: number[][], HH: number[][]): number[][] {
    const halfRows = LL.length;
    const halfCols = LL[0].length;
    const rows = halfRows * 2;
    const cols = halfCols * 2;

    const data = Array.from({ length: rows }, () => new Array(cols).fill(0));

    for (let i = 0; i < halfRows; i++) {
        for (let j = 0; j < halfCols; j++) {
            const ll = LL[i][j];
            const lh = LH[i][j];
            const hl = HL[i][j];
            const hh = HH[i][j];

            data[2 * i][2 * j] = (ll + lh + hl + hh) / 2;
            data[2 * i][2 * j + 1] = (ll - lh + hl - hh) / 2;
            data[2 * i + 1][2 * j] = (ll + lh - hl - hh) / 2;
            data[2 * i + 1][2 * j + 1] = (ll - lh - hl + hh) / 2;
        }
    }

    return data;
}

/**
 * Multi-level 2D Haar Discrete Wavelet Transform
 */
function dwtN(data: number[][], level: number): { LL: number[][], subbands: any[] } {
    let currentLL = data;
    const subbands = [];
    for (let i = 0; i < level; i++) {
        const result = dwt2(currentLL);
        subbands.push(result);
        currentLL = result.LL;
    }
    return { LL: currentLL, subbands };
}

/**
 * Multi-level Inverse 2D Haar Discrete Wavelet Transform
 */
function idwtN(LL: number[][], subbands: any[]): number[][] {
    let currentLL = LL;
    for (let i = subbands.length - 1; i >= 0; i--) {
        const { LH, HL, HH } = subbands[i];
        currentLL = idwt2(currentLL, LH, HL, HH);
    }
    return currentLL;
}

/**
 * Calculate PSNR (Peak Signal to Noise Ratio)
 */
function calculatePSNR(original: number[][], modified: number[][]): number {
    let mse = 0;
    const rows = original.length;
    const cols = original[0].length;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            mse += Math.pow(original[i][j] - modified[i][j], 2);
        }
    }
    mse /= (rows * cols);
    if (mse === 0) return 100;
    return 10 * Math.log10(Math.pow(255, 2) / mse);
}

/**
 * Calculate NC (Normalized Correlation) for binary watermarks
 */
function calculateNC(original: number[][], extracted: number[][]): number {
    const rows = original.length;
    const cols = original[0].length;
    let numerator = 0;
    let sumW2 = 0;
    let sumWp2 = 0;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const w = original[i][j] > 127 ? 1 : 0;
            const wp = extracted[i][j] > 127 ? 1 : 0;
            numerator += w * wp;
            sumW2 += w * w;
            sumWp2 += wp * wp;
        }
    }

    if (sumW2 === 0 || sumWp2 === 0) {
        // Fallback to bit-wise matching if one is all zeros to avoid division by zero
        let matches = 0;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const w = original[i][j] > 127 ? 1 : 0;
                const wp = extracted[i][j] > 127 ? 1 : 0;
                if (w === wp) matches++;
            }
        }
        return matches / (rows * cols);
    }
    
    return numerator / Math.sqrt(sumW2 * sumWp2);
}

/**
 * Calculate SSIM (Structural Similarity Index)
 */
function calculateSSIM(img1: number[][], img2: number[][]): number {
    const rows = img1.length;
    const cols = img1[0].length;
    
    let mu1 = 0, mu2 = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            mu1 += img1[i][j];
            mu2 += img2[i][j];
        }
    }
    mu1 /= (rows * cols);
    mu2 /= (rows * cols);

    let sigma1_sq = 0, sigma2_sq = 0, sigma12 = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            sigma1_sq += Math.pow(img1[i][j] - mu1, 2);
            sigma2_sq += Math.pow(img2[i][j] - mu2, 2);
            sigma12 += (img1[i][j] - mu1) * (img2[i][j] - mu2);
        }
    }
    sigma1_sq /= (rows * cols - 1);
    sigma2_sq /= (rows * cols - 1);
    sigma12 /= (rows * cols - 1);

    const C1 = Math.pow(0.01 * 255, 2);
    const C2 = Math.pow(0.03 * 255, 2);

    const numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2);
    const denominator = (Math.pow(mu1, 2) + Math.pow(mu2, 2) + C1) * (sigma1_sq + sigma2_sq + C2);

    return numerator / denominator;
}

/**
 * Genetic Algorithm for Optimization
 */
async function optimizeParameters(coverData: number[][], watermarkData: number[][], populationSize: number = 12, generations: number = 10, mutationRate: number = 0.2) {
    // For hybrid image watermarking, we optimize the quantization step Q
    // We want to maximize Q (for robustness) while keeping PSNR > 35 and SSIM > 0.95
    
    // Initial population: Q values between 10 and 150
    let population = Array.from({ length: populationSize }, () => Math.random() * 140 + 10);
    
    let bestIndividual = { Q: 40, fitness: -Infinity };

    for (let g = 0; g < generations; g++) {
        const results = population.map(Q => {
            // 1. Estimate Invisibility (PSNR & SSIM)
            // Empirical model for DWT-SVD QIM:
            // PSNR decreases logarithmically with Q
            // SSIM decreases linearly with Q
            const estimatedPSNR = 60 - 18 * Math.log10(Q);
            const estimatedSSIM = 1.0 - (Q / 800);
            
            // 2. Estimate Robustness
            // Robustness increases with Q
            const estimatedRobustness = Q / 200; // Normalized 0-1 range for Q up to 200
            
            // 3. Multi-objective Fitness Function
            // We use a weighted sum with penalties for low quality
            let fitness = 0;
            
            // Weights
            const w_psnr = 1.5;
            const w_ssim = 100; // SSIM is on a small scale (0-1)
            const w_robust = 50;
            
            // Base fitness from robustness
            fitness += estimatedRobustness * w_robust;
            
            // Quality rewards/penalties
            if (estimatedPSNR > 38) {
                fitness += (estimatedPSNR - 38) * w_psnr;
            } else if (estimatedPSNR < 35) {
                // Heavy penalty for low PSNR
                fitness -= (35 - estimatedPSNR) * 100;
            }
            
            if (estimatedSSIM > 0.98) {
                fitness += (estimatedSSIM - 0.98) * w_ssim;
            } else if (estimatedSSIM < 0.95) {
                // Heavy penalty for low SSIM
                fitness -= (0.95 - estimatedSSIM) * 500;
            }

            return { Q, fitness };
        });

        // Sort by fitness
        results.sort((a, b) => b.fitness - a.fitness);

        // Update best
        if (results[0].fitness > bestIndividual.fitness) {
            bestIndividual = { ...results[0] };
        }

        // Selection: Elitism (keep top 2) + Tournament Selection for the rest
        const newPopulation = [results[0].Q, results[1].Q];
        
        while (newPopulation.length < populationSize) {
            // Tournament selection
            const idx1 = Math.floor(Math.random() * populationSize);
            const idx2 = Math.floor(Math.random() * populationSize);
            const parent1 = results[idx1].fitness > results[idx2].fitness ? results[idx1].Q : results[idx2].Q;
            
            const idx3 = Math.floor(Math.random() * populationSize);
            const idx4 = Math.floor(Math.random() * populationSize);
            const parent2 = results[idx3].fitness > results[idx4].fitness ? results[idx3].Q : results[idx4].Q;
            
            // Crossover (Arithmetic)
            let child = (parent1 + parent2) / 2;
            
            // Mutation (Gaussian)
            if (Math.random() < mutationRate) {
                child += (Math.random() - 0.5) * 30;
            }
            
            // Clamp Q
            child = Math.max(5, Math.min(200, child));
            newPopulation.push(child);
        }
        
        population = newPopulation;
    }

    return { Q: bestIndividual.Q };
}

// --- API Endpoints ---

app.post("/api/embed", upload.fields([{ name: 'cover' }, { name: 'watermark' }]), async (req: any, res) => {
    try {
        if (!req.files) {
            return res.status(400).json({ error: "No files were uploaded. Please select a cover image and a watermark image." });
        }
        if (!req.files['cover']) {
            return res.status(400).json({ error: "Missing cover image file. Please upload a cover image." });
        }
        if (!req.files['watermark']) {
            return res.status(400).json({ error: "Missing watermark image file. Please upload a watermark image." });
        }

        const coverBuffer = req.files['cover'][0].buffer;
        const watermarkBuffer = req.files['watermark'][0].buffer;

        let coverImg, watermarkImg;
        try {
            coverImg = await Jimp.read(coverBuffer);
        } catch (e) {
            return res.status(400).json({ error: "Invalid cover image format. Please upload a valid JPG, PNG, or BMP file for the cover." });
        }

        try {
            watermarkImg = await Jimp.read(watermarkBuffer);
        } catch (e) {
            return res.status(400).json({ error: "Invalid watermark image format. Please upload a valid JPG, PNG, or BMP file for the watermark." });
        }

        const coverWidth = coverImg.width;
        const coverHeight = coverImg.height;

        if (coverWidth < 64 || coverHeight < 64) {
            return res.status(400).json({ error: `Cover image is too small (${coverWidth}x${coverHeight}). Minimum size is 64x64 pixels for the hybrid image watermarking process.` });
        }

        const watermarkWidth = watermarkImg.width;
        const watermarkHeight = watermarkImg.height;
        if (watermarkWidth < 4 || watermarkHeight < 4) {
            return res.status(400).json({ error: `Watermark image is too small (${watermarkWidth}x${watermarkHeight}). Minimum size is 4x4 pixels.` });
        }

        // Ensure square and dimensions are multiples of 8 for DWT/SVD blocks
        const size = Math.floor(Math.min(coverWidth, coverHeight) / 8) * 8;
        await coverImg.resize({ w: size, h: size });

        const finalWidth = coverImg.width;
        const finalHeight = coverImg.height;

        const dwtLevel = req.body.dwtLevel ? parseInt(req.body.dwtLevel) : 1;
        const blockSize = req.body.blockSize ? parseInt(req.body.blockSize) : 4;
        const gaPopulationSize = req.body.gaPopulationSize ? parseInt(req.body.gaPopulationSize) : 12;
        const gaGenerations = req.body.gaGenerations ? parseInt(req.body.gaGenerations) : 10;
        const gaMutationRate = req.body.gaMutationRate ? parseFloat(req.body.gaMutationRate) : 0.2;

        const llSize = finalWidth / Math.pow(2, dwtLevel);
        const numBlocks = Math.floor(llSize / blockSize); 

        if (numBlocks < 1) {
            return res.status(400).json({ error: `DWT Level ${dwtLevel} and Block Size ${blockSize} are too large for this image size. Resulting embedding space is zero.` });
        }

        // Resize watermark to fit the number of blocks (numBlocks x numBlocks)
        await watermarkImg.resize({ w: numBlocks, h: numBlocks });
        
        const opacity = req.body.opacity ? parseFloat(req.body.opacity) : 100;
        // Apply opacity by blending with white background
        if (opacity < 100) {
            const whiteBg = new Jimp({ width: numBlocks, height: numBlocks, color: 0xFFFFFFFF });
            watermarkImg.opacity(opacity / 100);
            whiteBg.composite(watermarkImg, 0, 0);
            watermarkImg = whiteBg;
        }

        await watermarkImg.greyscale();

        // Extract Y, Cb, Cr channels from cover
        const yChannel: number[][] = [];
        const cbChannel: number[][] = [];
        const crChannel: number[][] = [];

        for (let y = 0; y < finalHeight; y++) {
            const yRow = [];
            const cbRow = [];
            const crRow = [];
            for (let x = 0; x < finalWidth; x++) {
                const { r, g, b } = intToRGBA(coverImg.getPixelColor(x, y));
                const Y = rgbToY(r, g, b);
                const Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
                const Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
                yRow.push(Y);
                cbRow.push(Cb);
                crRow.push(Cr);
            }
            yChannel.push(yRow);
            cbChannel.push(cbRow);
            crChannel.push(crRow);
        }

        const watermarkData: number[][] = Array.from({ length: numBlocks }, () => new Array(numBlocks).fill(0));
        for (let y = 0; y < numBlocks; y++) {
            for (let x = 0; x < numBlocks; x++) {
                // Binarize watermark for QIM
                const val = intToRGBA(watermarkImg.getPixelColor(x, y)).r;
                watermarkData[y][x] = val > 127 ? 1 : 0;
            }
        }

        // GA Optimization for Quantization Step Q or use provided alpha
        let Q: number;
        if (req.body.alpha) {
            Q = parseFloat(req.body.alpha);
            if (isNaN(Q) || Q <= 0) {
                return res.status(400).json({ error: "Invalid manual quantization step (Q). It must be a positive number." });
            }
        } else {
            const optimization = await optimizeParameters(yChannel, watermarkData, gaPopulationSize, gaGenerations, gaMutationRate);
            Q = optimization.Q;
        }

        // 1. DWT on Y Channel
        const { LL, subbands } = dwtN(yChannel, dwtLevel);

        // 2. Block-based SVD + QIM Embedding
        const LL_modified = Array.from({ length: LL.length }, () => new Array(LL[0].length).fill(0));
        
        for (let i = 0; i < numBlocks; i++) {
            for (let j = 0; j < numBlocks; j++) {
                // Extract block
                const blockData = [];
                for (let bi = 0; bi < blockSize; bi++) {
                    const row = [];
                    for (let bj = 0; bj < blockSize; bj++) {
                        row.push(LL[i * blockSize + bi][j * blockSize + bj]);
                    }
                    blockData.push(row);
                }

                const matrixB = new Matrix(blockData);
                const svdB = new SingularValueDecomposition(matrixB);
                const Ub = svdB.leftSingularVectors;
                const Sb = Matrix.diag(svdB.diagonal);
                const Vb = svdB.rightSingularVectors;

                // QIM on the largest singular value
                let s11 = Sb.get(0, 0);
                const bit = watermarkData[i][j];
                
                // Quantization formula
                if (bit === 1) {
                    s11 = Math.floor(s11 / Q) * Q + 0.75 * Q;
                } else {
                    s11 = Math.floor(s11 / Q) * Q + 0.25 * Q;
                }
                Sb.set(0, 0, s11);

                const blockModified = Ub.mmul(Sb).mmul(Vb.transpose()).to2DArray();
                for (let bi = 0; bi < blockSize; bi++) {
                    for (let bj = 0; bj < blockSize; bj++) {
                        LL_modified[i * blockSize + bi][j * blockSize + bj] = blockModified[bi][bj];
                    }
                }
            }
        }

        // Fill remaining LL if any
        for (let i = 0; i < LL.length; i++) {
            for (let j = 0; j < LL[0].length; j++) {
                if (i >= numBlocks * blockSize || j >= numBlocks * blockSize) {
                    LL_modified[i][j] = LL[i][j];
                }
            }
        }

        // 6. IDWT
        const reconstructedY = idwtN(LL_modified, subbands);

        // Create Result Image (Reconstruct RGB from Y'CbCr)
        const resultImg = new Jimp({ width: finalWidth, height: finalHeight });
        for (let y = 0; y < finalHeight; y++) {
            for (let x = 0; x < finalWidth; x++) {
                const Y = reconstructedY[y][x];
                const Cb = cbChannel[y][x];
                const Cr = crChannel[y][x];
                const { r, g, b } = yCbCrToRgb(Y, Cb, Cr);
                resultImg.setPixelColor(rgbaToInt(r, g, b, 255), x, y);
            }
        }

        const psnr = calculatePSNR(yChannel, reconstructedY);
        const ssim = calculateSSIM(yChannel, reconstructedY);
        const base64 = await resultImg.getBase64("image/png");

        // Generate sub-band previews (from level 1 for visualization)
        const l1 = subbands[0];
        const llPreview = await subbandToBase64(l1.LL);
        const lhPreview = await subbandToBase64(l1.LH);
        const hlPreview = await subbandToBase64(l1.HL);
        const hhPreview = await subbandToBase64(l1.HH);

        res.json({
            image: base64,
            previews: {
                LL: llPreview,
                LH: lhPreview,
                HL: hlPreview,
                HH: hhPreview
            },
            metrics: {
                psnr: psnr.toFixed(2),
                ssim: ssim.toFixed(4),
                alpha: Q.toFixed(2) // We use Q as the parameter now
            }
        });

    } catch (error: any) {
        console.error("Embedding Error:", error);
        res.status(500).json({ error: error.message || "Watermarking process failed" });
    }
});

app.post("/api/extract", upload.fields([{ name: 'watermarked' }, { name: 'reference' }]), async (req: any, res) => {
    try {
        if (!req.files || !req.files['watermarked']) {
            return res.status(400).json({ error: "Missing watermarked image file. Please upload the image you wish to extract the watermark from." });
        }

        const alphaRaw = req.body.alpha;
        if (alphaRaw === undefined || alphaRaw === null || alphaRaw === '') {
            return res.status(400).json({ error: "Missing quantization step (Q) parameter. This is required for extraction." });
        }

        const Q = parseFloat(alphaRaw);
        if (isNaN(Q) || Q <= 0) {
            return res.status(400).json({ error: "Invalid quantization step (Q). It must be a positive number (e.g., 40)." });
        }

        const dwtLevel = req.body.dwtLevel ? parseInt(req.body.dwtLevel) : 1;
        const blockSize = req.body.blockSize ? parseInt(req.body.blockSize) : 4;

        const watermarkedBuffer = req.files['watermarked'][0].buffer;
        const referenceBuffer = req.files['reference'] ? req.files['reference'][0].buffer : null;

        let watermarkedImg;
        try {
            watermarkedImg = await Jimp.read(watermarkedBuffer);
        } catch (e) {
            return res.status(400).json({ error: "Invalid watermarked image format. Please upload a valid JPG, PNG, or BMP file." });
        }

        let referenceImg;
        if (referenceBuffer) {
            try {
                referenceImg = await Jimp.read(referenceBuffer);
            } catch (e) {
                // Ignore invalid reference image
            }
        }

        const originalWidth = watermarkedImg.width;
        const originalHeight = watermarkedImg.height;

        if (originalWidth < 64 || originalHeight < 64) {
            return res.status(400).json({ error: `Watermarked image is too small (${originalWidth}x${originalHeight}). Minimum size is 64x64 pixels for hybrid image watermark extraction.` });
        }

        // Ensure square and dimensions are multiples of 8 matching embedding
        const size = Math.floor(Math.min(originalWidth, originalHeight) / 8) * 8;
        await watermarkedImg.resize({ w: size, h: size });

        const width = size;
        const height = size;

        // Extract Y channel
        const yW: number[][] = [];

        for (let y = 0; y < height; y++) {
            const rowW = [];
            for (let x = 0; x < width; x++) {
                const pW = intToRGBA(watermarkedImg.getPixelColor(x, y));
                rowW.push(rgbToY(pW.r, pW.g, pW.b));
            }
            yW.push(rowW);
        }

        // 1. DWT on Y channel
        const { LL, subbands } = dwtN(yW, dwtLevel);

        // Generate sub-band previews for extraction (from level 1 for visualization)
        const l1 = subbands[0];
        const llPreview = await subbandToBase64(l1.LL);
        const lhPreview = await subbandToBase64(l1.LH);
        const hlPreview = await subbandToBase64(l1.HL);
        const hhPreview = await subbandToBase64(l1.HH);

        const llSize = LL.length;
        const numBlocks = Math.floor(llSize / blockSize);

        if (numBlocks < 1) {
            return res.status(400).json({ error: `DWT Level ${dwtLevel} and Block Size ${blockSize} are too large for this image size. Extraction space is zero.` });
        }

        // 2. Hybrid Image Extraction using QIM
        const extractedImg = new Jimp({ width: numBlocks, height: numBlocks });

        for (let i = 0; i < numBlocks; i++) {
            for (let j = 0; j < numBlocks; j++) {
                // Extract block
                const blockData = [];
                for (let bi = 0; bi < blockSize; bi++) {
                    const row = [];
                    for (let bj = 0; bj < blockSize; bj++) {
                        row.push(LL[i * blockSize + bi][j * blockSize + bj]);
                    }
                    blockData.push(row);
                }

                const matrixB = new Matrix(blockData);
                const svdB = new SingularValueDecomposition(matrixB);
                const s11 = svdB.diagonal[0];

                // QIM Decision
                const rem = s11 % Q;
                const val = rem > (Q / 2) ? 255 : 0;
                
                extractedImg.setPixelColor(rgbaToInt(val, val, val, 255), j, i);
            }
        }

        const base64 = await extractedImg.getBase64("image/png");

        let nc = null;
        if (referenceImg) {
            await referenceImg.resize({ w: numBlocks, h: numBlocks });
            const refData: number[][] = [];
            const extData: number[][] = [];
            
            for (let y = 0; y < numBlocks; y++) {
                const refRow = [];
                const extRow = [];
                for (let x = 0; x < numBlocks; x++) {
                    const pRef = intToRGBA(referenceImg.getPixelColor(x, y));
                    refRow.push(rgbToY(pRef.r, pRef.g, pRef.b));
                    
                    const pExt = intToRGBA(extractedImg.getPixelColor(x, y));
                    extRow.push(pExt.r);
                }
                refData.push(refRow);
                extData.push(extRow);
            }
            nc = calculateNC(refData, extData);
        }

        res.json({
            image: base64,
            nc: nc !== null ? nc.toFixed(4) : null,
            previews: {
                LL: llPreview,
                LH: lhPreview,
                HL: hlPreview,
                HH: hhPreview
            },
            message: "Watermark extracted using hybrid image techniques (no original image needed)"
        });

    } catch (error: any) {
        console.error("Extraction Error:", error);
        res.status(500).json({ error: error.message || "Watermark extraction failed" });
    }
});

app.post("/api/analyze", upload.single('image'), async (req: any, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No image provided" });
        }
        const img = await Jimp.read(req.file.buffer);
        
        const width = img.width;
        const height = img.height;
        const size = Math.floor(Math.min(width, height) / 8) * 8;
        await img.resize({ w: size, h: size });

        const finalWidth = img.width;
        const finalHeight = img.height;

        const yChannel: number[][] = [];
        for (let y = 0; y < finalHeight; y++) {
            const row = [];
            for (let x = 0; x < finalWidth; x++) {
                const p = intToRGBA(img.getPixelColor(x, y));
                row.push(rgbToY(p.r, p.g, p.b));
            }
            yChannel.push(row);
        }
        const { subbands } = dwtN(yChannel, 1);
        const l1 = subbands[0];
        res.json({
            previews: {
                LL: await subbandToBase64(l1.LL),
                LH: await subbandToBase64(l1.LH),
                HL: await subbandToBase64(l1.HL),
                HH: await subbandToBase64(l1.HH)
            }
        });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/attack", upload.fields([{ name: 'image' }]), async (req: any, res) => {
    try {
        if (!req.files || !req.files['image']) {
            return res.status(400).json({ error: "Missing image for attack simulation." });
        }

        const type = req.body.type;
        if (!type) {
            return res.status(400).json({ error: "Missing attack type." });
        }

        const intensity = parseFloat(req.body.intensity);
        if (isNaN(intensity)) {
            return res.status(400).json({ error: "Invalid or missing attack intensity." });
        }

        const buffer = req.files['image'][0].buffer;
        let img;
        try {
            img = await Jimp.read(buffer);
        } catch (e) {
            return res.status(400).json({ error: "Invalid image format for attack simulation." });
        }

        switch (type) {
            case 'compression':
                const quality = Math.max(1, Math.min(100, Math.round(100 - intensity * 90)));
                const jpegBuffer = await img.getBuffer("image/jpeg", { quality });
                img = await Jimp.read(jpegBuffer);
                break;
            case 'noise':
                // Salt and Pepper noise
                for (let y = 0; y < img.height; y++) {
                    for (let x = 0; x < img.width; x++) {
                        if (Math.random() < intensity * 0.1) {
                            const val = Math.random() > 0.5 ? 255 : 0;
                            img.setPixelColor(rgbaToInt(val, val, val, 255), x, y);
                        }
                    }
                }
                break;
            case 'blur':
                // Gaussian Blur
                const radius = Math.max(1, Math.floor(intensity * 5));
                await img.blur(radius);
                break;
            case 'median':
                // Median Filter (simulated with a small blur for robustness testing)
                await img.blur(Math.max(1, Math.floor(intensity * 3)));
                break;
            case 'rotation':
                const deg = intensity * 15; // Max 15 degrees
                await img.rotate(deg);
                break;
            case 'cropping':
                const cropPercent = intensity * 0.3; // Max 30%
                const w = img.width;
                const h = img.height;
                const cw = Math.floor(w * (1 - cropPercent));
                const ch = Math.floor(h * (1 - cropPercent));
                const cx = Math.floor((w - cw) / 2);
                const cy = Math.floor((h - ch) / 2);
                await img.crop({ x: cx, y: cy, w: cw, h: ch });
                await img.resize({ w, h }); 
                break;
            case 'sharpen':
                await img.convolute([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]);
                break;
            case 'brightness':
                await img.brightness(intensity);
                break;
            case 'contrast':
                await img.contrast(intensity);
                break;
            case 'scaling':
                const scale = 1 - intensity * 0.5;
                const sw = Math.floor(img.width * scale);
                const sh = Math.floor(img.height * scale);
                await img.resize({ w: sw, h: sh });
                await img.resize({ w: img.width, h: img.height });
                break;
        }

        const base64 = await img.getBase64("image/png");
        res.json({ image: base64 });

    } catch (error: any) {
        console.error("Attack Error:", error);
        res.status(500).json({ error: error.message || "Attack simulation failed" });
    }
});

app.post("/api/robustness", upload.fields([{ name: 'originalWatermark' }, { name: 'extractedWatermark' }]), async (req: any, res) => {
    try {
        if (!req.files || !req.files['originalWatermark'] || !req.files['extractedWatermark']) {
            return res.status(400).json({ error: "Missing watermarks for comparison. Both original and extracted watermarks are required." });
        }

        let owImg, ewImg;
        try {
            owImg = await Jimp.read(req.files['originalWatermark'][0].buffer);
        } catch (e) {
            return res.status(400).json({ error: "Invalid original watermark format for robustness comparison." });
        }

        try {
            ewImg = await Jimp.read(req.files['extractedWatermark'][0].buffer);
        } catch (e) {
            return res.status(400).json({ error: "Invalid extracted watermark format for robustness comparison." });
        }

        // Ensure same size
        if (ewImg.width !== owImg.width || ewImg.height !== owImg.height) {
            await ewImg.resize({ w: owImg.width, h: owImg.height });
        }

        const owData: number[][] = [];
        const ewData: number[][] = [];

        for (let y = 0; y < owImg.height; y++) {
            const rowO = [];
            const rowE = [];
            for (let x = 0; x < owImg.width; x++) {
                rowO.push(intToRGBA(owImg.getPixelColor(x, y)).r);
                rowE.push(intToRGBA(ewImg.getPixelColor(x, y)).r);
            }
            owData.push(rowO);
            ewData.push(rowE);
        }

        const nc = calculateNC(owData, ewData);
        res.json({ score: nc.toFixed(4) });

    } catch (error: any) {
        console.error("Robustness Calc Error:", error);
        res.status(500).json({ error: error.message || "Robustness calculation failed" });
    }
});

// --- Error Handling ---
app.use((err: any, req: any, res: any, next: any) => {
    console.error("Global Error:", err);
    res.status(500).json({ error: err.message || "Internal Server Error" });
});

// Catch-all for undefined API routes to prevent HTML fallback
app.all("/api/*", (req, res) => {
    res.status(404).json({ error: `API route ${req.method} ${req.url} not found` });
});

// --- Vite Middleware ---

async function startServer() {
    if (process.env.NODE_ENV !== "production" && !process.env.VERCEL) {
        const viteModule = "vite";
        const { createServer: createViteServer } = await import(viteModule);
        const vite = await createViteServer({
            server: { middlewareMode: true },
            appType: "spa",
        });
        app.use(vite.middlewares);
    } else {
        const distPath = path.join(process.cwd(), 'dist');
        app.use(express.static(distPath));
        app.get('*', (req, res) => {
            res.sendFile(path.join(distPath, 'index.html'));
        });
    }

    if (!process.env.VERCEL) {
        app.listen(PORT, "0.0.0.0", () => {
            console.log(`Server running on http://localhost:${PORT}`);
        });
    }
}

if (!process.env.VERCEL) {
    startServer();
}

export default app;
