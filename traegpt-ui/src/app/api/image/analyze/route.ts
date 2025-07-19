import { NextRequest, NextResponse } from 'next/server';

interface ImageClassificationResult {
  label: string;
  score: number;
}

export async function POST(req: NextRequest) {
  const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
  // Use the proper image classification model with the new API key
  const HF_API_URL = 'https://api-inference.huggingface.co/models/google/vit-base-patch16-224';
  
  if (!HF_API_KEY) {
    console.error('[Image Analyze] HuggingFace API key not set');
    return NextResponse.json({ error: 'HuggingFace API key not set on server.' }, { status: 500 });
  }
  
  console.log('[Image Analyze] API key found, starting analysis...');
  console.log('[Image Analyze] API key length:', HF_API_KEY.length);
  console.log('[Image Analyze] API key starts with:', HF_API_KEY.substring(0, 10) + '...');
  
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      console.error('[Image Analyze] No file provided');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    console.log('[Image Analyze] Processing file:', file.name, 'size:', file.size, 'type:', file.type);
    
    // Convert file to buffer
    const buffer = Buffer.from(await file.arrayBuffer());
    console.log('[Image Analyze] File converted to buffer, size:', buffer.length);
    
    // Add timeout to the fetch request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
    
    const start = Date.now();
    console.log('[Image Analyze] Making request to HuggingFace API...');
    
    const response = await fetch(HF_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/octet-stream',
      },
      body: buffer,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    const elapsed = Date.now() - start;
    
    console.log('[Image Analyze] Response status:', response.status, 'elapsed:', elapsed + 'ms');
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Image Analyze] HuggingFace API error:', response.status, errorText);
      
      return NextResponse.json({ 
        error: 'Image analysis failed', 
        detail: errorText,
        status: response.status 
      }, { status: response.status });
    }
    
    const data = await response.json() as ImageClassificationResult[];
    console.log('[Image Analyze] Success, elapsed:', elapsed + 'ms, data length:', data?.length || 0);
    console.log('[Image Analyze] Raw data sample:', JSON.stringify(data?.slice(0, 3)));
    
    // Process the image classification output to create a text description
    let imageDescription = "I can see in this image: ";
    
    if (data && Array.isArray(data) && data.length > 0) {
      const detectedObjects = data.slice(0, 5).map((item: ImageClassificationResult) => {
        const label = item.label || 'object';
        const score = item.score ? ` (${(item.score * 100).toFixed(1)}% confidence)` : '';
        return label + score;
      });
      
      imageDescription += detectedObjects.join(", ") + ".";
    } else {
      imageDescription += "various objects and elements.";
    }
    
    console.log('[Image Analyze] Final description:', imageDescription);
    
    return NextResponse.json({
      description: imageDescription,
      raw_data: data,
      analysis_time: elapsed / 1000
    });
    
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      console.error('[Image Analyze] Request timeout');
      return NextResponse.json({ error: 'Image analysis timeout' }, { status: 408 });
    }
    console.error('[Image Analyze] Unexpected error:', err);
    return NextResponse.json({ 
      error: 'Image analysis failed', 
      detail: String(err) 
    }, { status: 500 });
  }
} 