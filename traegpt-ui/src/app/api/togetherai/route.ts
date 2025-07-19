import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;
  const TOGETHER_API_URL = process.env.TOGETHER_API_URL || 'https://api.together.xyz/v1/chat/completions';
  
  if (!TOGETHER_API_KEY) {
    return NextResponse.json({ error: 'TogetherAI API key not set on server.' }, { status: 500 });
  }
  
  try {
    const { messages } = await req.json();
    
    // Add timeout to the fetch request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
    
    const togetherRes = await fetch(TOGETHER_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TOGETHER_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'moonshotai/kimi-k2-instruct',
        messages,
        max_tokens: 1000, // Limit response length for faster responses
        temperature: 0.7, // Slightly lower for more focused responses
        stream: false, // Disable streaming for faster single response
      }),
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!togetherRes.ok) {
      const errorText = await togetherRes.text();
      console.error('TogetherAI API error:', togetherRes.status, errorText);
      return NextResponse.json({ 
        error: 'TogetherAI API error', 
        detail: errorText,
        status: togetherRes.status 
      }, { status: togetherRes.status });
    }
    
    const data = await togetherRes.json();
    return NextResponse.json(data, { status: togetherRes.status });
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      return NextResponse.json({ error: 'Request timeout' }, { status: 408 });
    }
    console.error('TogetherAI proxy error:', err);
    return NextResponse.json({ 
      error: 'TogetherAI proxy error', 
      detail: String(err) 
    }, { status: 500 });
  }
} 