import { NextRequest, NextResponse } from 'next/server';

export const config = {
  api: {
    bodyParser: false,
  },
};

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    if (!file) {
      console.error('No file uploaded');
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }
    console.log('File received:', file.name, file.type, file.size);
    // Convert file to buffer
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    // Choose a public image classification model (e.g., google/vit-base-patch16-224)
    const model = 'google/vit-base-patch16-224';
    const hfRes = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY || ''}`,
        'Content-Type': 'application/octet-stream',
      },
      body: buffer,
    });
    console.log('HuggingFace response status:', hfRes.status);
    if (!hfRes.ok) {
      const err = await hfRes.text();
      console.error('HuggingFace API error:', err);
      return NextResponse.json({ error: 'HuggingFace API error', detail: err }, { status: 500 });
    }
    const data = await hfRes.json();
    console.log('HuggingFace API result:', data);
    return NextResponse.json(data);
  } catch (err) {
    console.error('API route error:', err);
    return NextResponse.json({ error: 'Internal server error', detail: String(err) }, { status: 500 });
  }
} 