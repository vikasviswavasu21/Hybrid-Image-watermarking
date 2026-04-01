// Hybrid Image Protection Frontend - v1.0.1
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, Download, RefreshCw, CheckCircle2, AlertCircle, Info, BarChart3, Settings2, ShieldAlert, Zap, Scissors, RotateCw, Wind, Sun, Contrast, Maximize } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface Metrics {
  psnr: string;
  ssim: string;
  alpha: string;
  dwtLevel?: number;
  blockSize?: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'embed' | 'extract'>('embed');
  const [gaStep, setGaStep] = useState<string>('');
  const [gaProgress, setGaProgress] = useState<number>(0);
  const [gaStepIndex, setGaStepIndex] = useState<number>(-1);
  
  // Embedding State
  const [coverImage, setCoverImage] = useState<File | null>(null);
  const [watermarkImage, setWatermarkImage] = useState<File | null>(null);
  const [coverPreview, setCoverPreview] = useState<string | null>(null);
  const [watermarkPreview, setWatermarkPreview] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attacking, setAttacking] = useState<string | null>(null);
  const [attackedImage, setAttackedImage] = useState<string | null>(null);
  const [robustnessScore, setRobustnessScore] = useState<string | null>(null);
  const [attackResults, setAttackResults] = useState<{ name: string, score: number }[]>([]);
  const [manualAlpha, setManualAlpha] = useState<number>(60);
  const [watermarkOpacity, setWatermarkOpacity] = useState<number>(100);
  const [dwtLevel, setDwtLevel] = useState<number>(1);
  const [blockSize, setBlockSize] = useState<number>(4);
  const [gaPopulationSize, setGaPopulationSize] = useState<number>(12);
  const [gaGenerations, setGaGenerations] = useState<number>(10);
  const [gaMutationRate, setGaMutationRate] = useState<number>(0.2);
  
  const gaMessages = [
    "Initializing Population (Random Q values)",
    "Fitness Evaluation: PSNR & Robustness Check",
    "Selection: Picking Best Candidates",
    "Crossover: Combining Genetic Material",
    "Mutation: Introducing Random Variations",
    `Generation 1/${gaGenerations}: Evolving...`,
    `Generation ${Math.floor(gaGenerations / 2)}/${gaGenerations}: Optimizing...`,
    `Generation ${gaGenerations}/${gaGenerations}: Converging...`,
    "Finalizing Optimal Parameters",
    "Applying Watermark with Best Q"
  ];
  
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [subbandPreviews, setSubbandPreviews] = useState<{ LL: string, LH: string, HL: string, HH: string } | null>(null);
  const [originalSubbandPreviews, setOriginalSubbandPreviews] = useState<{ LL: string, LH: string, HL: string, HH: string } | null>(null);
  const debounceTimer = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Extracting State
  const [watermarkedFile, setWatermarkedFile] = useState<File | null>(null);
  const [watermarkedPreview, setWatermarkedPreview] = useState<string | null>(null);
  const [extractedImage, setExtractedImage] = useState<string | null>(null);
  const [referenceWatermark, setReferenceWatermark] = useState<File | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);
  const [extractionNc, setExtractionNc] = useState<string | null>(null);
  const [extractSubbandPreviews, setExtractSubbandPreviews] = useState<{ LL: string, LH: string, HL: string, HH: string } | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [extractionAlpha, setExtractionAlpha] = useState<string>('40');
  const [extractDwtLevel, setExtractDwtLevel] = useState<number>(1);
  const [extractBlockSize, setExtractBlockSize] = useState<number>(4);

  // Pre-processing State
  const [embedQuality, setEmbedQuality] = useState(90);
  const [embedResolution, setEmbedResolution] = useState('720p');
  const [extractQuality, setExtractQuality] = useState(100);
  const [extractResolution, setExtractResolution] = useState('original');
  const [rawCover, setRawCover] = useState<File | null>(null);
  const [rawWatermark, setRawWatermark] = useState<File | null>(null);
  const [rawWatermarked, setRawWatermarked] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const processImage = async (file: File, quality: number, resolution: string): Promise<{ file: File, preview: string }> => {
    if (resolution === 'original' && quality === 100) {
      return { file, preview: URL.createObjectURL(file) };
    }
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;

        if (resolution === '480p') {
          const scale = Math.min(854 / width, 480 / height, 1);
          width *= scale;
          height *= scale;
        } else if (resolution === '720p') {
          const scale = Math.min(1280 / width, 720 / height, 1);
          width *= scale;
          height *= scale;
        } else if (resolution === '1080p') {
          const scale = Math.min(1920 / width, 1080 / height, 1);
          width *= scale;
          height *= scale;
        } else if (resolution === '2K') {
          const scale = Math.min(2560 / width, 1440 / height, 1);
          width *= scale;
          height *= scale;
        } else if (resolution === '4K') {
          const scale = Math.min(3840 / width, 2160 / height, 1);
          width *= scale;
          height *= scale;
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(img, 0, 0, width, height);

        canvas.toBlob((blob) => {
          if (blob) {
            const newFile = new File([blob], file.name, { type: 'image/jpeg' });
            resolve({ file: newFile, preview: URL.createObjectURL(newFile) });
          }
        }, 'image/jpeg', quality / 100);
      };
      img.src = URL.createObjectURL(file);
    });
  };

  const onDropCover = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setRawCover(file);
    setIsProcessing(true);
    const processed = await processImage(file, embedQuality, embedResolution);
    setCoverImage(processed.file);
    setCoverPreview(processed.preview);
    
    // Analyze cover for DWT
    const formData = new FormData();
    formData.append('image', processed.file);
    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        setOriginalSubbandPreviews(data.previews);
      }
    } catch (e) {
      console.error("Failed to analyze cover image", e);
    }

    setIsProcessing(false);
    setResultImage(null);
    setMetrics(null);
  }, [embedQuality, embedResolution]);

  const onDropWatermark = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setRawWatermark(file);
    setWatermarkImage(file);
    setWatermarkPreview(URL.createObjectURL(file));
    setResultImage(null);
    setMetrics(null);
  }, []);

  const onDropWatermarked = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setRawWatermarked(file);
    setIsProcessing(true);
    const processed = await processImage(file, extractQuality, extractResolution);
    setWatermarkedFile(processed.file);
    setWatermarkedPreview(processed.preview);
    setIsProcessing(false);
    setExtractedImage(null);
    setExtractionNc(null);
  }, [extractQuality, extractResolution]);

  const onDropReference = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    const preview = URL.createObjectURL(file);
    setReferenceWatermark(file);
    setReferencePreview(preview);
    setExtractionNc(null);
  }, []);

  // Re-process when settings change
  useEffect(() => {
    const update = async () => {
      if (rawCover) {
        setIsProcessing(true);
        const processed = await processImage(rawCover, embedQuality, embedResolution);
        setCoverImage(processed.file);
        setCoverPreview(processed.preview);
        setIsProcessing(false);
      }
    };
    update();
  }, [embedQuality, embedResolution]);

  useEffect(() => {
    const update = async () => {
      if (rawWatermarked) {
        setIsProcessing(true);
        const processed = await processImage(rawWatermarked, extractQuality, extractResolution);
        setWatermarkedFile(processed.file);
        setWatermarkedPreview(processed.preview);
        setIsProcessing(false);
      }
    };
    update();
  }, [extractQuality, extractResolution]);

  const { getRootProps: getCoverRootProps, getInputProps: getCoverInputProps, isDragActive: isCoverDragActive } = useDropzone({
    onDrop: onDropCover,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp'] },
    multiple: false
  } as any);

  const { getRootProps: getWatermarkRootProps, getInputProps: getWatermarkInputProps, isDragActive: isWatermarkDragActive } = useDropzone({
    onDrop: onDropWatermark,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp'] },
    multiple: false
  } as any);

  const { getRootProps: getWatermarkedRootProps, getInputProps: getWatermarkedInputProps, isDragActive: isWatermarkedDragActive } = useDropzone({
    onDrop: onDropWatermarked,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp'] },
    multiple: false
  } as any);

  const { getRootProps: getReferenceRootProps, getInputProps: getReferenceInputProps, isDragActive: isReferenceDragActive } = useDropzone({
    onDrop: onDropReference,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp'] },
    multiple: false
  } as any);

  const [isStressTesting, setIsStressTesting] = useState(false);

  const runAllAttacks = async () => {
    if (!resultImage || !watermarkImage || !metrics) return;
    
    setIsStressTesting(true);
    setError(null);
    
    try {
      const resBlob = await fetch(resultImage).then(r => r.blob());
      const wmBlob = await fetch(watermarkPreview!).then(r => r.blob());
      
      const formData = new FormData();
      formData.append('image', resBlob, 'watermarked.png');
      formData.append('watermark', wmBlob, 'watermark.png');
      formData.append('alpha', metrics.alpha.toString());
      formData.append('dwtLevel', (metrics.dwtLevel || dwtLevel).toString());
      formData.append('blockSize', (metrics.blockSize || blockSize).toString());
      
      const attackList = [
        { id: 'compression', intensity: 0.4 },
        { id: 'noise', intensity: 0.1 },
        { id: 'blur', intensity: 0.3 },
        { id: 'median', intensity: 0.2 },
        { id: 'rotation', intensity: 0.1 },
        { id: 'cropping', intensity: 0.1 },
        { id: 'sharpen', intensity: 0.4 },
        { id: 'brightness', intensity: 0.2 },
        { id: 'contrast', intensity: 0.2 },
        { id: 'scaling', intensity: 0.3 }
      ];
      
      formData.append('attacks', JSON.stringify(attackList));
      
      const response = await fetch('/api/stress-test', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Stress test failed');
        } else {
          const text = await response.text();
          console.error("Non-JSON error response:", text);
          throw new Error(`Server error (${response.status}). Please check server logs.`);
        }
      }
      
      const data = await response.json();
      setAttackResults(data.results);
      
      // Calculate average score
      const avg = data.results.reduce((acc: number, curr: any) => acc + curr.score, 0) / data.results.length;
      setRobustnessScore(avg.toFixed(4));
      
    } catch (err: any) {
      setError(err.message || 'Stress test failed');
      console.error(err);
    } finally {
      setIsStressTesting(false);
    }
  };

  const handleAttack = async (type: 'compression' | 'noise' | 'rotation' | 'cropping' | 'blur' | 'median' | 'sharpen' | 'brightness' | 'contrast' | 'scaling', intensity: number) => {
    if (!resultImage || !watermarkImage || !metrics) return;

    setAttacking(type);
    setRobustnessScore(null);
    
    try {
      const resBlob = await fetch(resultImage).then(r => r.blob());
      const wmBlob = await fetch(watermarkPreview!).then(r => r.blob());
      
      const formData = new FormData();
      formData.append('image', resBlob, 'watermarked.png');
      formData.append('watermark', wmBlob, 'watermark.png');
      formData.append('alpha', metrics.alpha.toString());
      formData.append('dwtLevel', (metrics.dwtLevel || dwtLevel).toString());
      formData.append('blockSize', (metrics.blockSize || blockSize).toString());
      formData.append('attacks', JSON.stringify([{ id: type, intensity }]));

      const response = await fetch('/api/stress-test', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Attack simulation failed');
        } else {
          const text = await response.text();
          console.error("Non-JSON error response:", text);
          throw new Error(`Server error (${response.status}). Please check server logs.`);
        }
      }
      
      const data = await response.json();
      const result = data.results[0];
      
      setRobustnessScore(result.score.toFixed(4));
      
      // Update attack results for the graph
      setAttackResults(prev => {
        const existing = prev.findIndex(r => r.name === type);
        const newResult = { name: type, score: result.score };
        if (existing !== -1) {
          const updated = [...prev];
          updated[existing] = newResult;
          return updated;
        }
        return [...prev, newResult];
      });

    } catch (err: any) {
      setError(err.message || 'Attack simulation failed');
      console.error(err);
    } finally {
      setAttacking(null);
    }
  };

  const handleProcess = async (isManual: boolean = false) => {
    if (!coverImage || !watermarkImage) return;

    if (!isManual) {
      setLoading(true);
      setGaProgress(0);
      setGaStepIndex(0);
      setGaStep(gaMessages[0]);
      
      // Simulate GA steps for visual feedback
      let stepIdx = 0;
      const stepInterval = setInterval(() => {
        stepIdx++;
        if (stepIdx < gaMessages.length) {
          setGaStep(gaMessages[stepIdx]);
          setGaStepIndex(stepIdx);
          setGaProgress((stepIdx / gaMessages.length) * 100);
        } else {
          clearInterval(stepInterval);
        }
      }, 600);
      
      (window as any)._gaInterval = stepInterval;
    } else {
      setIsPreviewing(true);
    }
    
    setError(null);
    
    // Cancel previous preview request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const formData = new FormData();
    formData.append('cover', coverImage);
    formData.append('watermark', watermarkImage);
    formData.append('opacity', watermarkOpacity.toString());
    formData.append('dwtLevel', dwtLevel.toString());
    formData.append('blockSize', blockSize.toString());
    formData.append('gaPopulationSize', gaPopulationSize.toString());
    formData.append('gaGenerations', gaGenerations.toString());
    formData.append('gaMutationRate', gaMutationRate.toString());
    if (isManual) {
      formData.append('alpha', manualAlpha.toString());
    }

    try {
      const response = await fetch('/api/embed', {
        method: 'POST',
        body: formData,
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Embedding failed' }));
        throw new Error(errorData.error || 'Embedding failed');
      }

      const data = await response.json();

      setResultImage(data.image);
      setSubbandPreviews(data.previews);
      setMetrics(data.metrics);
      setAttackResults([]); // Reset attack results for new image
      if (data.metrics?.alpha && !isManual) {
        setExtractionAlpha(data.metrics.alpha);
        setManualAlpha(parseFloat(data.metrics.alpha));
        setGaProgress(100);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      setError(err.message || 'An error occurred during watermarking. Please try again.');
      console.error(err);
    } finally {
      if (!isManual) {
        setLoading(false);
        if ((window as any)._gaInterval) clearInterval((window as any)._gaInterval);
      }
      else setIsPreviewing(false);
    }
  };

  // Real-time preview effect
  useEffect(() => {
    if (!coverImage || !watermarkImage || loading) return;

    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      handleProcess(true);
    }, 300);

    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, [manualAlpha, watermarkOpacity, dwtLevel, blockSize, coverImage, watermarkImage]);

  const handleExtract = async () => {
    if (!watermarkedFile) return;

    setExtracting(true);
    setError(null);
    const formData = new FormData();
    formData.append('watermarked', watermarkedFile);
    formData.append('alpha', extractionAlpha);
    formData.append('dwtLevel', extractDwtLevel.toString());
    formData.append('blockSize', extractBlockSize.toString());
    
    // Use the reference watermark if provided, otherwise fallback to the original watermark from the embed tab
    const ref = referenceWatermark || watermarkImage;
    if (ref) {
      formData.append('reference', ref);
    }

    try {
      const response = await fetch('/api/extract', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Extraction failed' }));
        throw new Error(errorData.error || 'Extraction failed');
      }

      const data = await response.json();

      setExtractedImage(data.image);
      setExtractionNc(data.nc);
      setExtractSubbandPreviews(data.previews);
    } catch (err: any) {
      setError(err.message || 'An error occurred during extraction. Please try again.');
      console.error(err);
    } finally {
      setExtracting(false);
    }
  };

  const downloadResult = () => {
    if (!resultImage) return;
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = 'watermarked_image.png';
    link.click();
  };

  const downloadExtracted = () => {
    if (!extractedImage) return;
    const link = document.createElement('a');
    link.href = extractedImage;
    link.download = 'extracted_watermark.png';
    link.click();
  };

  const downloadReport = () => {
    if (!metrics) return;
    const report = `
HYBRID IMAGE PROTECTION - ANALYSIS REPORT
==========================================
Date: ${new Date().toLocaleString()}
Algorithm: DWT-SVD with Genetic Algorithm Optimization

METRICS:
--------
Peak Signal to Noise Ratio (PSNR): ${metrics.psnr} dB
Structural Similarity Index (SSIM): ${metrics.ssim}
Optimized Scaling Factor (Alpha): ${metrics.alpha}
Wavelet Transform: 2D Haar DWT (LL, LH, HL, HH sub-bands generated)

SUMMARY:
--------
The PSNR value of ${metrics.psnr} dB indicates ${parseFloat(metrics.psnr) > 35 ? 'excellent' : 'good'} visual quality of the watermarked image.
An SSIM of ${metrics.ssim} confirms high structural preservation compared to the original image.
The Discrete Wavelet Transform (DWT) was used to decompose the image into frequency sub-bands, with the watermark embedded in the SVD-transformed LL sub-band for optimal robustness.
The Genetic Algorithm successfully converged on an optimal alpha of ${metrics.alpha} to balance invisibility and robustness.

This report confirms the integrity and security of the watermarked asset.
`;
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'watermarking_analysis_report.txt';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#0F172A] text-slate-100 font-sans selection:bg-indigo-500/30">
      {/* Animated Background Gradient */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[40%] -left-[20%] w-[80%] h-[80%] rounded-full bg-indigo-600/10 blur-[120px] animate-pulse"></div>
        <div className="absolute -bottom-[40%] -right-[20%] w-[80%] h-[80%] rounded-full bg-blue-600/10 blur-[120px] animate-pulse delay-700"></div>
      </div>

      {/* Header */}
      <header className="bg-slate-900/50 backdrop-blur-xl border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <ShieldAlert className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">Hybrid <span className="text-indigo-400">Image Protection</span></h1>
              <p className="text-[10px] text-slate-500 font-medium uppercase tracking-widest">DWT-SVD QIM Optimization</p>
            </div>
          </div>
          
          <div className="flex bg-slate-900/50 p-1.5 rounded-2xl border border-slate-800/50 backdrop-blur-md">
            <button 
              onClick={() => setActiveTab('embed')}
              className={`px-8 py-2.5 rounded-xl text-sm font-bold transition-all duration-500 relative ${activeTab === 'embed' ? 'text-white' : 'text-slate-500 hover:text-slate-300'}`}
            >
              {activeTab === 'embed' && (
                <motion.div layoutId="tab-bg" className="absolute inset-0 bg-indigo-600 rounded-xl shadow-lg shadow-indigo-600/20" />
              )}
              <span className="relative z-10">Embed</span>
            </button>
            <button 
              onClick={() => setActiveTab('extract')}
              className={`px-8 py-2.5 rounded-xl text-sm font-bold transition-all duration-500 relative ${activeTab === 'extract' ? 'text-white' : 'text-slate-500 hover:text-slate-300'}`}
            >
              {activeTab === 'extract' && (
                <motion.div layoutId="tab-bg" className="absolute inset-0 bg-indigo-600 rounded-xl shadow-lg shadow-indigo-600/20" />
              )}
              <span className="relative z-10">Extract</span>
            </button>
          </div>

          <div className="hidden md:flex items-center gap-4">
            <div className="h-8 w-[1px] bg-slate-800 mx-2"></div>
            <button className="text-sm font-semibold text-slate-400 hover:text-white transition-colors">Docs</button>
            <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-semibold transition-all border border-slate-700">GitHub</button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative">
        
        {/* Hero Section */}
        <section className="text-center mb-16 space-y-8 relative">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-indigo-500/5 blur-[100px] -z-10"></div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-indigo-500/10 to-blue-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-black uppercase tracking-[0.3em] shadow-xl shadow-indigo-500/5"
          >
            <RefreshCw className="w-3.5 h-3.5 animate-spin-slow" />
            Next-Gen Digital Watermarking
          </motion.div>
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-5xl md:text-7xl font-black tracking-tighter text-white leading-[1.1]"
          >
            Protect Your Assets with <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 via-blue-400 to-cyan-400 animate-gradient-x">Intelligent Security</span>
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="max-w-2xl mx-auto text-slate-400 text-lg md:text-xl font-medium leading-relaxed"
          >
            Advanced Hybrid Image DWT-SVD protection using QIM optimized by Genetic Algorithms. 
            <span className="text-indigo-400/80"> No original image needed for extraction.</span>
          </motion.p>
        </section>

        {/* Algorithm Explanation Cards */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {[
            { 
              title: "DWT Decomposition", 
              desc: "Splits image into frequency sub-bands (LL, LH, HL, HH) to find the most stable areas for embedding.",
              icon: <BarChart3 className="w-6 h-6 text-indigo-400" />,
              color: "from-indigo-500/20 to-transparent"
            },
            { 
              title: "SVD Transformation", 
              desc: "Modifies singular values of the LL sub-band, providing high capacity and resistance to geometric attacks.",
              icon: <ImageIcon className="w-6 h-6 text-blue-400" />,
              color: "from-blue-500/20 to-transparent"
            },
            { 
              title: "Hybrid Image Extraction", 
              desc: "Uses Quantization Index Modulation (QIM) to extract the watermark without needing the original cover image.",
              icon: <Zap className="w-6 h-6 text-cyan-400" />,
              color: "from-cyan-500/20 to-transparent"
            }
          ].map((item, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + i * 0.1 }}
              className={`p-8 rounded-3xl bg-slate-900/50 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all`}
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${item.color} opacity-0 group-hover:opacity-100 transition-opacity`}></div>
              <div className="relative z-10 space-y-4">
                <div className="w-12 h-12 rounded-2xl bg-slate-800 flex items-center justify-center border border-slate-700">
                  {item.icon}
                </div>
                <h3 className="text-xl font-bold text-white">{item.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{item.desc}</p>
              </div>
            </motion.div>
          ))}
        </section>

        {/* About Section */}
        <section className="mb-16 bg-slate-900/40 backdrop-blur-sm rounded-3xl border border-slate-800/50 p-8">
          <div className="flex items-center gap-3 mb-4">
            <Info className="w-5 h-5 text-indigo-400" />
            <h3 className="text-lg font-bold text-white">About Hybrid Image Protection</h3>
          </div>
          <p className="text-sm text-slate-400 leading-relaxed">
            This project implements a state-of-the-art <strong>Hybrid Image Digital Watermarking</strong> system. By combining 
            <strong> Discrete Wavelet Transform (DWT)</strong> and <strong> Singular Value Decomposition (SVD)</strong> with 
            <strong> Quantization Index Modulation (QIM)</strong>, we ensure that your watermarks can be extracted 
            without needing the original image. The <strong> Genetic Algorithm (GA)</strong> automatically optimizes 
            the quantization step (Q) to achieve the perfect balance between image quality (PSNR) and watermark robustness.
          </p>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Uploads & Controls */}
          <div className="lg:col-span-5 space-y-6">
            <AnimatePresence mode="wait">
              <motion.section 
                key={activeTab}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-2xl"
              >
                <div className="flex items-center gap-3 mb-8">
                  <div className="p-2 bg-indigo-500/10 rounded-lg">
                    <Upload className="w-5 h-5 text-indigo-400" />
                  </div>
                  <h2 className="text-xl font-bold text-white">
                    {activeTab === 'embed' ? 'Hybrid Image Embedding Engine' : 'Hybrid Image Extraction Engine'}
                  </h2>
                </div>

                {/* Pre-processing Settings */}
                {activeTab === 'embed' && (
                  <div className="bg-slate-950/50 border border-slate-800 rounded-2xl p-5 mb-8 space-y-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Settings2 className="w-4 h-4 text-indigo-400" />
                        <h3 className="text-[10px] font-black text-white uppercase tracking-[0.2em]">Pre-processing Controls</h3>
                      </div>
                      {isProcessing && (
                        <div className="flex items-center gap-2 text-[9px] font-bold text-indigo-400 uppercase animate-pulse">
                          <RefreshCw className="w-3 h-3 animate-spin" />
                          Processing...
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">JPEG Quality</label>
                          <span className="text-[10px] font-bold text-indigo-400">
                            {embedQuality}%
                          </span>
                        </div>
                        <input 
                          type="range" 
                          min="1" 
                          max="100" 
                          value={embedQuality}
                          onChange={(e) => setEmbedQuality(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>

                      <div className="space-y-3">
                        <label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Target Resolution</label>
                        <select 
                          value={embedResolution}
                          onChange={(e) => setEmbedResolution(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs font-bold text-slate-300 outline-none focus:border-indigo-500/50 transition-all"
                        >
                          <option value="original">Original Size</option>
                          <option value="4K">4K (UHD)</option>
                          <option value="2K">2K (QHD)</option>
                          <option value="1080p">1080p (FHD)</option>
                          <option value="720p">720p (HD)</option>
                          <option value="480p">480p (SD)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}
                
                {activeTab === 'extract' && isProcessing && (
                  <div className="flex items-center gap-2 text-[9px] font-bold text-indigo-400 uppercase animate-pulse mb-8">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    Processing Image...
                  </div>
                )}
                
                {activeTab === 'embed' ? (
                  <div className="space-y-6">
                    {/* Cover Image Upload */}
                    <div className="space-y-3">
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Original Cover Image</label>
                      <div 
                        {...getCoverRootProps()} 
                        className={`relative aspect-video rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden flex flex-col items-center justify-center gap-3
                          ${isCoverDragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'}`}
                      >
                        <input {...getCoverInputProps()} />
                        {coverPreview ? (
                          <img src={coverPreview} className="absolute inset-0 w-full h-full object-cover" alt="Cover preview" />
                        ) : (
                          <>
                            <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800">
                              <ImageIcon className="w-6 h-6 text-slate-600" />
                            </div>
                            <p className="text-xs text-slate-500 font-medium">Drop cover image here</p>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Watermark Image Upload */}
                    <div className="space-y-3">
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Watermark Logo</label>
                      <div 
                        {...getWatermarkRootProps()} 
                        className={`relative aspect-video rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden flex flex-col items-center justify-center gap-3
                          ${isWatermarkDragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'}`}
                      >
                        <input {...getWatermarkInputProps()} />
                        {watermarkPreview ? (
                          <img src={watermarkPreview} className="absolute inset-0 w-full h-full object-contain p-6" alt="Watermark preview" />
                        ) : (
                          <>
                            <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800">
                              <ImageIcon className="w-6 h-6 text-slate-600" />
                            </div>
                            <p className="text-xs text-slate-500 font-medium">Drop logo/signature here</p>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Scaling Factor (Alpha) Slider */}
                    <div className="space-y-4 pt-4 border-t border-slate-800/50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Scaling Factor (Q)</label>
                          {isPreviewing && <RefreshCw className="w-3 h-3 text-indigo-400 animate-spin" />}
                        </div>
                        <span className="text-xs font-bold text-indigo-400">{manualAlpha.toFixed(1)}</span>
                      </div>
                      <input 
                        type="range" 
                        min="1" 
                        max="200" 
                        step="0.5"
                        value={manualAlpha}
                        disabled={loading}
                        onChange={(e) => setManualAlpha(parseFloat(e.target.value))}
                        className={`w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      />
                      <div className="flex justify-between text-[8px] text-slate-600 font-bold uppercase tracking-widest">
                        <span>More Invisible</span>
                        <span>More Robust</span>
                      </div>
                    </div>

                    {/* Watermark Opacity Slider */}
                    <div className="space-y-4 pt-4 border-t border-slate-800/50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Watermark Opacity</label>
                          {isPreviewing && <RefreshCw className="w-3 h-3 text-cyan-400 animate-spin" />}
                        </div>
                        <span className="text-xs font-bold text-cyan-400">{watermarkOpacity}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="100" 
                        step="1"
                        value={watermarkOpacity}
                        disabled={loading}
                        onChange={(e) => setWatermarkOpacity(parseInt(e.target.value))}
                        className={`w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      />
                      <div className="flex justify-between text-[8px] text-slate-600 font-bold uppercase tracking-widest">
                        <span>Transparent</span>
                        <span>Opaque</span>
                      </div>
                    </div>

                    {/* Advanced Embedding Settings */}
                    <div className="space-y-4 pt-4 border-t border-slate-800/50">
                      <div className="flex items-center gap-2 mb-2">
                        <Settings2 className="w-3 h-3 text-indigo-400" />
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Advanced Embedding Parameters</h4>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">DWT Level</label>
                            <span className="text-[9px] font-black text-indigo-400">{dwtLevel}</span>
                          </div>
                          <select 
                            value={dwtLevel}
                            onChange={(e) => setDwtLevel(parseInt(e.target.value))}
                            className="w-full bg-slate-950/50 border border-slate-800 rounded-lg py-2 px-3 text-[11px] font-bold text-white focus:border-indigo-500/50 outline-none transition-all"
                          >
                            <option value={1}>Level 1 (Standard)</option>
                            <option value={2}>Level 2 (Deep)</option>
                            <option value={3}>Level 3 (Very Deep)</option>
                          </select>
                        </div>

                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">SVD Block Size</label>
                            <span className="text-[9px] font-black text-indigo-400">{blockSize}x{blockSize}</span>
                          </div>
                          <select 
                            value={blockSize}
                            onChange={(e) => setBlockSize(parseInt(e.target.value))}
                            className="w-full bg-slate-950/50 border border-slate-800 rounded-lg py-2 px-3 text-[11px] font-bold text-white focus:border-indigo-500/50 outline-none transition-all"
                          >
                            <option value={2}>2x2 Blocks</option>
                            <option value={4}>4x4 Blocks (Standard)</option>
                            <option value={8}>8x8 Blocks</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Genetic Algorithm Optimization Parameters */}
                    <div className="space-y-4 pt-4 border-t border-slate-800/50">
                      <div className="flex items-center gap-2 mb-2">
                        <Zap className="w-3 h-3 text-yellow-400" />
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Genetic Algorithm Optimization</h4>
                      </div>
                      
                      <div className="grid grid-cols-1 gap-4">
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-1.5">
                              <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Population Size</label>
                              <div className="group relative">
                                <Info className="w-2.5 h-2.5 text-slate-600 cursor-help" />
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-900 border border-slate-800 rounded-lg text-[8px] text-slate-400 leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                                  Number of candidate Q values in each generation. Larger populations explore more space but are slower.
                                </div>
                              </div>
                            </div>
                            <span className="text-[9px] font-black text-yellow-400">{gaPopulationSize}</span>
                          </div>
                          <input 
                            type="range" 
                            min="4" 
                            max="50" 
                            step="2"
                            value={gaPopulationSize}
                            onChange={(e) => setGaPopulationSize(parseInt(e.target.value))}
                            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                          />
                        </div>

                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-1.5">
                              <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Generations</label>
                              <div className="group relative">
                                <Info className="w-2.5 h-2.5 text-slate-600 cursor-help" />
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-900 border border-slate-800 rounded-lg text-[8px] text-slate-400 leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                                  Number of iterations for the evolution process. More generations lead to better convergence.
                                </div>
                              </div>
                            </div>
                            <span className="text-[9px] font-black text-yellow-400">{gaGenerations}</span>
                          </div>
                          <input 
                            type="range" 
                            min="5" 
                            max="50" 
                            step="5"
                            value={gaGenerations}
                            onChange={(e) => setGaGenerations(parseInt(e.target.value))}
                            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                          />
                        </div>

                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-1.5">
                              <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Mutation Rate</label>
                              <div className="group relative">
                                <Info className="w-2.5 h-2.5 text-slate-600 cursor-help" />
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-900 border border-slate-800 rounded-lg text-[8px] text-slate-400 leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                                  Probability of random changes in Q values. Helps avoid local optima.
                                </div>
                              </div>
                            </div>
                            <span className="text-[9px] font-black text-yellow-400">{(gaMutationRate * 100).toFixed(0)}%</span>
                          </div>
                          <input 
                            type="range" 
                            min="0.05" 
                            max="0.5" 
                            step="0.05"
                            value={gaMutationRate}
                            onChange={(e) => setGaMutationRate(parseFloat(e.target.value))}
                            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Original Cover Analysis */}
                    {originalSubbandPreviews && (
                      <div className="space-y-4 pt-6 border-t border-slate-800/50">
                        <div className="flex items-center gap-2 mb-2">
                          <BarChart3 className="w-3 h-3 text-cyan-400" />
                          <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Original Cover DWT Analysis</h4>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          {[
                            { id: 'LL', label: 'LL', sub: 'Approx' },
                            { id: 'LH', label: 'LH', sub: 'Horiz' },
                            { id: 'HL', label: 'HL', sub: 'Vert' },
                            { id: 'HH', label: 'HH', sub: 'Diag' }
                          ].map((band) => (
                            <div key={band.id} className="bg-slate-950/50 p-2 rounded-xl border border-slate-800 space-y-1">
                              <div className="flex justify-between items-center">
                                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest">{band.label}</span>
                                <span className="text-[7px] text-slate-600 font-bold uppercase">{band.sub}</span>
                              </div>
                              <div className="aspect-square rounded-lg overflow-hidden bg-slate-900 border border-slate-800/50">
                                <img src={originalSubbandPreviews[band.id as keyof typeof originalSubbandPreviews]} className="w-full h-full object-cover" alt={band.label} />
                              </div>
                            </div>
                          ))}
                        </div>
                        <p className="text-[9px] text-slate-600 font-medium italic">This shows the frequency decomposition of your original cover image before embedding.</p>
                      </div>
                    )}

                    <button
                      onClick={() => handleProcess(false)}
                      disabled={!coverImage || !watermarkImage || loading || isPreviewing}
                      className={`w-full mt-8 py-4 px-6 rounded-2xl font-bold flex flex-col items-center justify-center gap-2 transition-all duration-300 relative overflow-hidden
                        ${!coverImage || !watermarkImage || loading || isPreviewing
                          ? 'bg-slate-800 text-slate-600 cursor-not-allowed' 
                          : 'bg-gradient-to-r from-indigo-600 to-blue-600 text-white hover:shadow-xl hover:shadow-indigo-500/20 active:scale-[0.98]'}`}
                    >
                      {loading && (
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${gaProgress}%` }}
                          className="absolute bottom-0 left-0 h-1 bg-white/30 z-0"
                        />
                      )}
                      <div className="flex items-center gap-3 relative z-10">
                        {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Settings2 className="w-5 h-5" />}
                        <span>{loading ? 'Optimizing Parameters...' : 'Run GA Optimization'}</span>
                      </div>
                      {loading && (
                        <motion.span 
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          key={gaStep}
                          className="text-[9px] font-black text-white/50 uppercase tracking-widest relative z-10"
                        >
                          {gaStep}
                        </motion.span>
                      )}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Watermarked Image Upload */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Watermarked Image</label>
                        <div 
                          {...getWatermarkedRootProps()} 
                          className={`relative aspect-square rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden flex flex-col items-center justify-center gap-3
                            ${isWatermarkedDragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'}`}
                        >
                          <input {...getWatermarkedInputProps()} />
                          {watermarkedPreview ? (
                            <img src={watermarkedPreview} className="absolute inset-0 w-full h-full object-cover" alt="Watermarked preview" />
                          ) : (
                            <>
                              <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800">
                                <ImageIcon className="w-6 h-6 text-slate-600" />
                              </div>
                              <p className="text-[10px] text-slate-500 font-medium text-center px-4">Drop image to extract from</p>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="space-y-3">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Reference Watermark (Optional)</label>
                        <div 
                          {...getReferenceRootProps()} 
                          className={`relative aspect-square rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden flex flex-col items-center justify-center gap-3
                            ${isReferenceDragActive ? 'border-cyan-500 bg-cyan-500/5' : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'}`}
                        >
                          <input {...getReferenceInputProps()} />
                          {referencePreview ? (
                            <img src={referencePreview} className="absolute inset-0 w-full h-full object-cover" alt="Reference preview" />
                          ) : (
                            <>
                              <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800">
                                <RefreshCw className="w-6 h-6 text-slate-600" />
                              </div>
                              <p className="text-[10px] text-slate-500 font-medium text-center px-4">
                                {watermarkPreview ? 'Using original watermark from Embed tab' : 'Drop original watermark for NC comparison'}
                              </p>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Alpha Parameter Input */}
                    <div className="space-y-3 mt-6">
                      <div className="flex items-center justify-between">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] block">Quantization Step (Q)</label>
                        <span className="text-[10px] font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded-full">Key for Extraction</span>
                      </div>
                      <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                          <Settings2 className="w-4 h-4 text-slate-500 group-focus-within:text-indigo-400 transition-colors" />
                        </div>
                        <input 
                          type="number" 
                          step="0.1"
                          value={extractionAlpha}
                          onChange={(e) => setExtractionAlpha(e.target.value)}
                          className="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-white focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all outline-none"
                          placeholder="e.g. 40"
                        />
                      </div>
                      <p className="text-[10px] text-slate-600 font-medium italic">Use the same Q value that was generated during embedding for best results.</p>
                    </div>

                    {/* Advanced Extraction Settings */}
                    <div className="space-y-4 pt-6 border-t border-slate-800/50">
                      <div className="flex items-center gap-2 mb-2">
                        <Settings2 className="w-3 h-3 text-cyan-400" />
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Advanced Extraction Parameters</h4>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">DWT Level</label>
                            <span className="text-[9px] font-black text-cyan-400">{extractDwtLevel}</span>
                          </div>
                          <select 
                            value={extractDwtLevel}
                            onChange={(e) => setExtractDwtLevel(parseInt(e.target.value))}
                            className="w-full bg-slate-950/50 border border-slate-800 rounded-lg py-2 px-3 text-[11px] font-bold text-white focus:border-cyan-500/50 outline-none transition-all"
                          >
                            <option value={1}>Level 1 (Standard)</option>
                            <option value={2}>Level 2 (Deep)</option>
                            <option value={3}>Level 3 (Very Deep)</option>
                          </select>
                        </div>

                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">SVD Block Size</label>
                            <span className="text-[9px] font-black text-cyan-400">{extractBlockSize}x{extractBlockSize}</span>
                          </div>
                          <select 
                            value={extractBlockSize}
                            onChange={(e) => setExtractBlockSize(parseInt(e.target.value))}
                            className="w-full bg-slate-950/50 border border-slate-800 rounded-lg py-2 px-3 text-[11px] font-bold text-white focus:border-cyan-500/50 outline-none transition-all"
                          >
                            <option value={2}>2x2 Blocks</option>
                            <option value={4}>4x4 Blocks (Standard)</option>
                            <option value={8}>8x8 Blocks</option>
                          </select>
                        </div>
                      </div>
                      <p className="text-[9px] text-slate-600 font-medium italic">Fine-tune these to match the embedding parameters if known, or experiment for recovery.</p>
                    </div>

                    <button
                      onClick={handleExtract}
                      disabled={!watermarkedFile || extracting}
                      className={`w-full mt-8 py-4 px-6 rounded-2xl font-bold flex items-center justify-center gap-3 transition-all duration-300
                        ${!watermarkedFile || extracting 
                          ? 'bg-slate-800 text-slate-600 cursor-not-allowed' 
                          : 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white hover:shadow-xl hover:shadow-cyan-500/20 active:scale-[0.98]'}`}
                    >
                      {extracting ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Settings2 className="w-5 h-5" />}
                      {extracting ? 'Decomposing Sub-bands...' : 'Extract Watermark'}
                    </button>
                  </div>
                )}
              </motion.section>
            </AnimatePresence>
          </div>

          {/* Right Column: Results & Metrics */}
          <div className="lg:col-span-7 space-y-6">
            <AnimatePresence mode="wait">
              {activeTab === 'embed' ? (
                loading ? (
                  <motion.div 
                    key="embed-loading"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="h-full min-h-[500px] flex flex-col items-center justify-center space-y-8 bg-slate-900/40 rounded-3xl border border-indigo-500/20 backdrop-blur-md"
                  >
                    <div className="relative">
                      <motion.div 
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        className="w-24 h-24 border-4 border-indigo-500/10 border-t-indigo-500 rounded-full"
                      />
                      <motion.div 
                        animate={{ rotate: -360 }}
                        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                        className="absolute inset-0 w-24 h-24 border-4 border-transparent border-b-blue-400 rounded-full scale-75"
                      />
                      <motion.div 
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="absolute inset-0 flex items-center justify-center"
                      >
                        <Settings2 className="w-8 h-8 text-indigo-400" />
                      </motion.div>
                    </div>
                    <div className="text-center space-y-3">
                      <h3 className="text-2xl font-black text-white tracking-tight">Optimizing Parameters</h3>
                      <p className="text-slate-400 font-medium max-w-[300px] mx-auto text-sm">Genetic Algorithm is finding the optimal scaling factor for maximum robustness.</p>
                      
                      <div className="mt-8 space-y-2 max-w-[320px] mx-auto text-left">
                        {gaMessages.map((msg, idx) => (
                          <motion.div 
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ 
                              opacity: idx <= gaStepIndex ? 1 : 0.3,
                              x: 0,
                              color: idx === gaStepIndex ? '#818cf8' : idx < gaStepIndex ? '#22c55e' : '#94a3b8'
                            }}
                            className="flex items-center gap-3 text-[11px] font-bold uppercase tracking-wider"
                          >
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center border ${idx < gaStepIndex ? 'bg-green-500/20 border-green-500/50' : idx === gaStepIndex ? 'bg-indigo-500/20 border-indigo-500/50' : 'bg-slate-800 border-slate-700'}`}>
                              {idx < gaStepIndex ? (
                                <CheckCircle2 className="w-2.5 h-2.5 text-green-500" />
                              ) : idx === gaStepIndex ? (
                                <RefreshCw className="w-2.5 h-2.5 text-indigo-400 animate-spin" />
                              ) : (
                                <div className="w-1 h-1 bg-slate-600 rounded-full" />
                              )}
                            </div>
                            <span>{msg}</span>
                          </motion.div>
                        ))}
                      </div>

                      <div className="flex justify-center gap-1 mt-8">
                        {[0, 1, 2].map(i => (
                          <motion.div 
                            key={i}
                            animate={{ y: [0, -6, 0] }}
                            transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.1 }}
                            className="w-1.5 h-1.5 bg-indigo-500 rounded-full"
                          />
                        ))}
                      </div>
                    </div>
                  </motion.div>
                ) : resultImage ? (
                  <motion.div
                    key="embed-result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="space-y-6"
                  >
                    {/* Result Image Card */}
                    <section className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-2xl">
                      <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-green-500/10 flex items-center justify-center">
                            <CheckCircle2 className="w-5 h-5 text-green-500" />
                          </div>
                          <h2 className="font-bold text-white">Comparison Analysis</h2>
                        </div>
                        <button 
                          onClick={downloadResult}
                          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-xl text-xs font-bold text-white transition-all shadow-lg shadow-indigo-500/20"
                        >
                          <Download className="w-4 h-4" />
                          Download Result
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-slate-800">
                        <div className="bg-slate-950 p-4 space-y-3">
                          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest text-center">Original Image</p>
                          <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-slate-800">
                            <img src={coverPreview!} className="max-w-full max-h-full object-contain" alt="Original" />
                          </div>
                        </div>
                        <div className="bg-slate-950 p-4 space-y-3 relative">
                          <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest text-center">Watermarked Result</p>
                          <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-indigo-500/20 ring-1 ring-indigo-500/10 relative">
                            <img src={resultImage} className={`max-w-full max-h-full object-contain transition-opacity duration-300 ${isPreviewing ? 'opacity-40' : 'opacity-100'}`} alt="Result" />
                            {isPreviewing && (
                              <div className="absolute inset-0 flex items-center justify-center bg-slate-950/20 backdrop-blur-[2px]">
                                <div className="flex flex-col items-center gap-2">
                                  <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin" />
                                  <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Updating Preview...</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </section>

                    {/* DWT Sub-band Analysis */}
                    {subbandPreviews && (
                      <section className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-2xl">
                        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-cyan-500/10 flex items-center justify-center">
                              <BarChart3 className="w-5 h-5 text-cyan-400" />
                            </div>
                            <h2 className="font-bold text-white">DWT Sub-band Analysis</h2>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-800">
                          {[
                            { id: 'LL', label: 'Low-Low (LL)', sub: 'Approximation' },
                            { id: 'LH', label: 'Low-High (LH)', sub: 'Horizontal Details' },
                            { id: 'HL', label: 'High-Low (HL)', sub: 'Vertical Details' },
                            { id: 'HH', label: 'High-High (HH)', sub: 'Diagonal Details' }
                          ].map((band) => (
                            <div key={band.id} className="bg-slate-950 p-4 space-y-3">
                              <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest text-center">{band.label}</p>
                              <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-slate-800">
                                <img src={subbandPreviews[band.id as keyof typeof subbandPreviews]} className="max-w-full max-h-full object-contain" alt={band.label} />
                              </div>
                              <p className="text-[8px] text-slate-600 text-center font-bold uppercase tracking-tighter">{band.sub}</p>
                            </div>
                          ))}
                        </div>
                      </section>
                    )}

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {[
                        { label: "PSNR", val: metrics?.psnr + " dB", sub: "Peak Signal to Noise", color: "text-indigo-400" },
                        { label: "SSIM", val: metrics?.ssim, sub: "Structural Similarity", color: "text-blue-400" },
                        { label: "Step (Q)", val: metrics?.alpha, sub: "GA Optimized Key", color: "text-cyan-400" }
                      ].map((m, i) => (
                        <div key={i} className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 hover:border-slate-700 transition-colors">
                          <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{m.label}</span>
                          <div className={`text-2xl font-black mt-1 ${m.color}`}>{m.val}</div>
                          <p className="text-[10px] text-slate-600 mt-1 font-medium">{m.sub}</p>
                        </div>
                      ))}
                    </div>

                    {/* Detailed Report Section */}
                    <motion.section 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden"
                    >
                      <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-indigo-500/10 flex items-center justify-center">
                            <BarChart3 className="w-5 h-5 text-indigo-400" />
                          </div>
                          <h2 className="font-bold text-white">Security Analysis Report</h2>
                        </div>
                        <button 
                          onClick={downloadReport}
                          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-xl text-xs font-bold text-white transition-all border border-slate-700 shadow-lg"
                        >
                          <Download className="w-4 h-4" />
                          Export Report (.txt)
                        </button>
                      </div>
                      <div className="p-8 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          <div className="space-y-4">
                            <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest">Visual Integrity</h4>
                            <p className="text-sm text-slate-400 leading-relaxed">
                              The Peak Signal to Noise Ratio (PSNR) of <span className="text-indigo-400 font-bold">{metrics?.psnr} dB</span> suggests that the watermark is perceptually invisible. 
                              Values above 35 dB are typically considered high quality for digital watermarking.
                            </p>
                          </div>
                          <div className="space-y-4">
                            <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest">Structural Preservation</h4>
                            <p className="text-sm text-slate-400 leading-relaxed">
                              An SSIM of <span className="text-blue-400 font-bold">{metrics?.ssim}</span> indicates that the structural information of the original image has been preserved with near-perfect accuracy. 
                              The Discrete Wavelet Transform (DWT) analysis above shows how the image energy is distributed across different frequency sub-bands.
                            </p>
                          </div>
                        </div>
                        <div className="pt-6 border-t border-slate-800">
                          <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4">Parameter Summary</h4>
                          <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800/50 text-xs text-slate-500 leading-relaxed">
                            {loading ? (
                              "Optimizing parameters for the best balance of quality and robustness..."
                            ) : metrics?.alpha === manualAlpha.toFixed(2) ? (
                              `The watermark is currently being applied with a manual quantization step (Q = ${metrics?.alpha}). This allows for direct control over the invisibility-robustness trade-off.`
                            ) : (
                              `The Genetic Algorithm has successfully converged on an optimal quantization step (Q = ${metrics?.alpha}). This parameter ensures that the watermark energy is maximized for robustness against attacks while maintaining the required invisibility threshold.`
                            )}
                          </div>
                        </div>
                      </div>
                    </motion.section>

                    {/* Attack Simulation Section */}
                    <motion.section 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden"
                    >
                      <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                        <div className="flex items-center justify-between w-full">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-red-500/10 flex items-center justify-center">
                              <ShieldAlert className="w-5 h-5 text-red-400" />
                            </div>
                            <h2 className="font-bold text-white">Robustness Stress Test</h2>
                          </div>
                          <button
                            onClick={runAllAttacks}
                            disabled={isStressTesting || attacking !== null}
                            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-black uppercase tracking-widest rounded-xl transition-all shadow-lg shadow-indigo-500/20"
                          >
                            {isStressTesting ? (
                              <>
                                <RefreshCw className="w-3 h-3 animate-spin" />
                                Testing...
                              </>
                            ) : (
                              <>
                                <Zap className="w-3 h-3" />
                                Run All Attacks
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                      <div className="p-8 space-y-8">
                        <p className="text-sm text-slate-400 leading-relaxed">
                          Simulate common image processing attacks to verify the algorithm's robustness. 
                          The <span className="text-white font-bold">Robustness Score (NC)</span> measures how well the watermark survives the attack (1.0 is perfect).
                        </p>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {[
                            { id: 'compression', label: 'JPEG Compression', icon: Download, intensity: 0.6 },
                            { id: 'noise', label: 'Gaussian Noise', icon: Zap, intensity: 0.1 },
                            { id: 'blur', label: 'Gaussian Blur', icon: Wind, intensity: 0.4 },
                            { id: 'median', label: 'Median Filter', icon: ShieldAlert, intensity: 0.3 },
                            { id: 'rotation', label: 'Small Rotation', icon: RotateCw, intensity: 0.3 },
                            { id: 'cropping', label: 'Center Cropping', icon: Scissors, intensity: 0.2 },
                            { id: 'sharpen', label: 'Sharpening', icon: Zap, intensity: 0.5 },
                            { id: 'brightness', label: 'Brightness', icon: Sun, intensity: 0.2 },
                            { id: 'contrast', label: 'Contrast', icon: Contrast, intensity: 0.3 },
                            { id: 'scaling', label: 'Scaling', icon: Maximize, intensity: 0.5 },
                          ].map((attack) => (
                            <button
                              key={attack.id}
                              onClick={() => handleAttack(attack.id as any, attack.intensity)}
                              disabled={attacking !== null}
                              className={`p-4 rounded-2xl border transition-all flex flex-col items-center gap-3 group
                                ${attacking === attack.id 
                                  ? 'bg-indigo-500/20 border-indigo-500/50' 
                                  : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 hover:bg-slate-900'}`}
                            >
                              <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors
                                ${attacking === attack.id ? 'bg-indigo-500 text-white' : 'bg-slate-900 text-slate-500 group-hover:text-slate-300'}`}>
                                {attacking === attack.id ? <RefreshCw className="w-5 h-5 animate-spin" /> : <attack.icon className="w-5 h-5" />}
                              </div>
                              <span className="text-[10px] font-black uppercase tracking-widest text-center">{attack.label}</span>
                            </button>
                          ))}
                        </div>

                        <AnimatePresence mode="wait">
                          {attackedImage && (
                            <motion.div 
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="pt-8 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-8 items-center"
                            >
                              <div className="space-y-4">
                                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Attacked Image Preview</h4>
                                <div className="aspect-video rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                                  <img src={attackedImage} className="max-w-full max-h-full object-contain" alt="Attacked" />
                                </div>
                              </div>
                              <div className="bg-slate-950/50 p-8 rounded-3xl border border-slate-800 flex flex-col items-center justify-center text-center gap-4">
                                <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Robustness Score (NC)</div>
                                {attacking ? (
                                  <div className="flex items-center gap-3 text-indigo-400 font-bold">
                                    <RefreshCw className="w-5 h-5 animate-spin" />
                                    <span>Calculating...</span>
                                  </div>
                                ) : (
                                  <>
                                    <div className={`text-6xl font-black ${parseFloat(robustnessScore || '0') > 0.9 ? 'text-green-400' : parseFloat(robustnessScore || '0') > 0.8 ? 'text-emerald-400' : parseFloat(robustnessScore || '0') > 0.6 ? 'text-yellow-500' : 'text-red-500'}`}>
                                      {robustnessScore}
                                    </div>
                                    <div className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[10px] font-black uppercase tracking-widest">
                                      {parseFloat(robustnessScore || '0') > 0.9 ? (
                                        <span className="text-green-400">Best</span>
                                      ) : parseFloat(robustnessScore || '0') > 0.8 ? (
                                        <span className="text-emerald-400">Excellent</span>
                                      ) : parseFloat(robustnessScore || '0') > 0.6 ? (
                                        <span className="text-yellow-500">Good</span>
                                      ) : (
                                        <span className="text-red-500">Poor</span>
                                      )}
                                    </div>
                                    <p className="text-xs text-slate-500 max-w-[200px]">
                                      {parseFloat(robustnessScore || '0') > 0.9 
                                        ? "Best robustness! The watermark is highly resilient to most common attacks." 
                                        : parseFloat(robustnessScore || '0') > 0.8
                                        ? "Excellent robustness! The watermark survives well under moderate stress."
                                        : parseFloat(robustnessScore || '0') > 0.6 
                                        ? "Good robustness. The watermark remains identifiable after standard processing."
                                        : "Poor robustness. The attack significantly impacted the watermark's integrity."}
                                    </p>
                                  </>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>

                        {/* Robustness Graph */}
                        {attackResults.length > 0 && (
                          <motion.div 
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="pt-8 border-t border-slate-800 space-y-6"
                          >
                            <div className="flex items-center justify-between">
                              <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Robustness Comparison Chart</h4>
                              <div className="flex items-center gap-4">
                                <div className="flex items-center gap-1.5">
                                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                                  <span className="text-[10px] text-slate-500 font-bold uppercase">High (NC &gt; 0.8)</span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                  <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                                  <span className="text-[10px] text-slate-500 font-bold uppercase">Moderate</span>
                                </div>
                              </div>
                            </div>
                            
                            <div className="h-[300px] w-full bg-slate-950/30 rounded-2xl p-4 border border-slate-800/50">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={[...attackResults].sort((a, b) => b.score - a.score)} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                  <XAxis 
                                    dataKey="name" 
                                    stroke="#64748b" 
                                    fontSize={10} 
                                    tickLine={false} 
                                    axisLine={false}
                                    tickFormatter={(val) => val.charAt(0).toUpperCase() + val.slice(1)}
                                  />
                                  <YAxis 
                                    stroke="#64748b" 
                                    fontSize={10} 
                                    tickLine={false} 
                                    axisLine={false} 
                                    domain={[0, 1]}
                                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                                  />
                                  <Tooltip 
                                    cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }}
                                    contentStyle={{ 
                                      backgroundColor: '#0f172a', 
                                      border: '1px solid #1e293b', 
                                      borderRadius: '12px',
                                      fontSize: '12px',
                                      fontWeight: 'bold',
                                      color: '#f1f5f9'
                                    }}
                                    itemStyle={{ color: '#818cf8' }}
                                    formatter={(value: number) => [value.toFixed(4), 'Robustness (NC)']}
                                    labelFormatter={(label) => label.charAt(0).toUpperCase() + label.slice(1) + ' Attack'}
                                  />
                                  <Bar dataKey="score" radius={[6, 6, 0, 0]} barSize={40}>
                                    {attackResults.map((entry, index) => (
                                      <Cell 
                                        key={`cell-${index}`} 
                                        fill={entry.score > 0.95 ? '#22c55e' : entry.score > 0.85 ? '#10b981' : entry.score > 0.7 ? '#eab308' : '#ef4444'} 
                                        fillOpacity={0.8}
                                      />
                                    ))}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="bg-indigo-500/5 border border-indigo-500/10 p-4 rounded-xl">
                                <h5 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-2">Algorithm Performance</h5>
                                <p className="text-[11px] text-slate-500 leading-relaxed">
                                  The hybrid DWT-SVD approach shows high resilience to frequency-domain attacks like JPEG compression and noise. Geometric attacks like rotation and cropping are handled through SVD's singular value stability.
                                </p>
                              </div>
                              <div className="bg-blue-500/5 border border-blue-500/10 p-4 rounded-xl">
                                <h5 className="text-[10px] font-black text-blue-400 uppercase tracking-widest mb-2">Optimization Insight</h5>
                                <p className="text-[11px] text-slate-500 leading-relaxed">
                                  {metrics?.alpha === manualAlpha.toFixed(2) 
                                    ? `The manual scaling factor (α = ${metrics?.alpha}) maintains an average robustness score of ${(attackResults.reduce((acc, curr) => acc + curr.score, 0) / attackResults.length).toFixed(4)} across all tested attack vectors.`
                                    : `The Genetic Algorithm has tuned the scaling factor (α = ${metrics?.alpha}) to maintain an average robustness score of ${(attackResults.reduce((acc, curr) => acc + curr.score, 0) / attackResults.length).toFixed(4)} across all tested attack vectors.`
                                  }
                                </p>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </motion.section>
                  </motion.div>
                ) : (
                  <div key="embed-empty" className="h-full min-h-[500px] bg-slate-900/30 rounded-3xl border-2 border-slate-800 border-dashed flex flex-col items-center justify-center text-slate-600 gap-6">
                    <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center border border-slate-800">
                      <ImageIcon className="w-10 h-10 opacity-10" />
                    </div>
                    <div className="text-center space-y-2">
                      <p className="text-lg font-bold text-slate-500">Awaiting Input</p>
                      <p className="text-sm text-slate-600 max-w-[280px]">Upload your images on the left to begin the intelligent watermarking process.</p>
                    </div>
                  </div>
                )
              ) : (
                extracting ? (
                  <motion.div 
                    key="extract-loading"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="h-full min-h-[500px] flex flex-col items-center justify-center space-y-8 bg-slate-900/40 rounded-3xl border border-cyan-500/20 backdrop-blur-md"
                  >
                    <div className="relative">
                      <motion.div 
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        className="w-24 h-24 border-4 border-cyan-500/10 border-t-cyan-500 rounded-full"
                      />
                      <motion.div 
                        animate={{ rotate: -360 }}
                        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                        className="absolute inset-0 w-24 h-24 border-4 border-transparent border-b-blue-400 rounded-full scale-75"
                      />
                      <motion.div 
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="absolute inset-0 flex items-center justify-center"
                      >
                        <RefreshCw className="w-8 h-8 text-cyan-400" />
                      </motion.div>
                    </div>
                    <div className="text-center space-y-3">
                      <h3 className="text-2xl font-black text-white tracking-tight">Extracting Watermark</h3>
                      <p className="text-slate-400 font-medium max-w-[300px] mx-auto">Decomposing frequency sub-bands and performing SVD analysis.</p>
                      <div className="flex justify-center gap-1 mt-4">
                        {[0, 1, 2].map(i => (
                          <motion.div 
                            key={i}
                            animate={{ y: [0, -6, 0] }}
                            transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.1 }}
                            className="w-1.5 h-1.5 bg-cyan-500 rounded-full"
                          />
                        ))}
                      </div>
                    </div>
                  </motion.div>
                ) : extractedImage ? (
                  <motion.div
                    key="extract-result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="space-y-6"
                  >
                    <section className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-2xl">
                      <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-cyan-500/10 flex items-center justify-center">
                            <CheckCircle2 className="w-5 h-5 text-cyan-500" />
                          </div>
                          <h2 className="font-bold text-white">Extraction Analysis</h2>
                        </div>
                        <button 
                          onClick={downloadExtracted}
                          className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-xl text-xs font-bold text-white transition-all shadow-lg shadow-cyan-500/20"
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </button>
                      </div>
                      <div className={`grid grid-cols-1 ${(referencePreview || watermarkPreview) ? 'md:grid-cols-3' : 'md:grid-cols-2'} gap-px bg-slate-800`}>
                        <div className="bg-slate-950 p-4 space-y-3">
                          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest text-center">Watermarked Input</p>
                          <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-slate-800">
                            <img src={watermarkedPreview!} className="max-w-full max-h-full object-contain" alt="Watermarked Input" />
                          </div>
                        </div>
                        
                        {(referencePreview || watermarkPreview) && (
                          <div className="bg-slate-950 p-4 space-y-3">
                            <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest text-center">
                              {referencePreview ? 'Reference Watermark' : 'Original Watermark'}
                            </p>
                            <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-slate-800">
                              <img src={referencePreview || watermarkPreview!} className="max-w-full max-h-full object-contain" alt="Reference Watermark" />
                            </div>
                          </div>
                        )}

                        <div className="bg-slate-950 p-4 space-y-3">
                          <p className="text-[10px] font-black text-cyan-400 uppercase tracking-widest text-center">Extracted Logo</p>
                          <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-cyan-500/20 ring-1 ring-cyan-500/10">
                            <div className="relative group">
                              <div className="absolute -inset-4 bg-cyan-500/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                              <img src={extractedImage} className="relative w-32 h-32 object-contain rounded-lg shadow-2xl" alt="Extracted Watermark" />
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {extractionNc && (
                        <div className="p-6 bg-slate-950 border-t border-slate-800 flex items-center justify-between">
                          <div className="space-y-1">
                            <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Normalized Correlation (NC)</p>
                            <div className="flex items-center gap-3">
                              <div className="text-2xl font-black text-cyan-400">{extractionNc}</div>
                              <div className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-[10px] font-bold text-cyan-400 uppercase tracking-wider">
                                {parseFloat(extractionNc) > 0.9 ? 'Excellent' : parseFloat(extractionNc) > 0.7 ? 'Good Match' : 'Poor Match'}
                              </div>
                            </div>
                          </div>
                          <div className="text-right hidden md:block">
                            <p className="text-[10px] text-slate-500 font-medium max-w-[200px]">
                              NC measures the similarity between the reference and extracted watermark. 1.0 is a perfect match.
                            </p>
                          </div>
                        </div>
                      )}
                    </section>

                    {/* DWT Sub-band Analysis (Extraction) */}
                    {extractSubbandPreviews && (
                      <section className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-2xl">
                        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-cyan-500/10 flex items-center justify-center">
                              <BarChart3 className="w-5 h-5 text-cyan-400" />
                            </div>
                            <h2 className="font-bold text-white">DWT Sub-band Analysis</h2>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-800">
                          {[
                            { id: 'LL', label: 'Low-Low (LL)', sub: 'Approximation' },
                            { id: 'LH', label: 'Low-High (LH)', sub: 'Horizontal Details' },
                            { id: 'HL', label: 'High-Low (HL)', sub: 'Vertical Details' },
                            { id: 'HH', label: 'High-High (HH)', sub: 'Diagonal Details' }
                          ].map((band) => (
                            <div key={band.id} className="bg-slate-950 p-4 space-y-3">
                              <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest text-center">{band.label}</p>
                              <div className="aspect-square flex items-center justify-center overflow-hidden rounded-xl bg-slate-900/50 border border-slate-800">
                                <img src={extractSubbandPreviews[band.id as keyof typeof extractSubbandPreviews]} className="max-w-full max-h-full object-contain" alt={band.label} />
                              </div>
                              <p className="text-[8px] text-slate-600 text-center font-bold uppercase tracking-tighter">{band.sub}</p>
                            </div>
                          ))}
                        </div>
                      </section>
                    )}
                  </motion.div>
                ) : (
                  <div key="extract-empty" className="h-full min-h-[500px] bg-slate-900/30 rounded-3xl border-2 border-slate-800 border-dashed flex flex-col items-center justify-center text-slate-600 gap-6">
                    <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center border border-slate-800">
                      <ImageIcon className="w-10 h-10 opacity-10" />
                    </div>
                    <div className="text-center space-y-2">
                      <p className="text-lg font-bold text-slate-500">Ready for Extraction</p>
                      <p className="text-sm text-slate-600 max-w-[280px]">Upload the watermarked image to recover the hidden logo. {watermarkPreview ? 'The original watermark from the Embed tab will be used for comparison.' : 'Provide a reference watermark for NC calculation.'}</p>
                    </div>
                  </div>
                )
              )}
            </AnimatePresence>

            {error && (
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-red-500/10 border border-red-500/20 p-6 rounded-2xl flex items-center gap-4 text-red-400 text-sm"
              >
                <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center shrink-0">
                  <AlertCircle className="w-6 h-6" />
                </div>
                <div>
                  <p className="font-bold">System Error</p>
                  <p className="opacity-80">{error}</p>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-24 border-t border-slate-800 bg-slate-900/50 backdrop-blur-xl py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-slate-800 rounded-lg flex items-center justify-center border border-slate-700">
                <ImageIcon className="text-slate-400 w-5 h-5" />
              </div>
              <span className="font-bold text-slate-300">Hybrid Image Protection</span>
            </div>
            <div className="flex items-center gap-8 text-xs font-bold text-slate-500 uppercase tracking-widest">
              <span className="hover:text-indigo-400 cursor-pointer transition-colors">DWT Transform</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">SVD Matrix</span>
              <span className="hover:text-cyan-400 cursor-pointer transition-colors">Genetic Algorithm</span>
            </div>
            <p className="text-xs text-slate-600 font-medium">© 2026 Digital Watermarking Research Lab</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
