import { NextRequest, NextResponse } from 'next/server';

interface GoogleSearchItem {
  title: string;
  link: string;
  snippet: string;
}

interface GoogleSearchResponse {
  items?: GoogleSearchItem[];
  searchInformation?: {
    totalResults: string;
    searchTime: number;
  };
}

export async function POST(req: NextRequest) {
  const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
  const GOOGLE_CSE_ID = process.env.GOOGLE_CSE_ID;
  
  if (!GOOGLE_API_KEY) {
    console.error('[Web Search] Missing GOOGLE_API_KEY environment variable');
    return NextResponse.json({ 
      error: 'Google API key not configured. Please set GOOGLE_API_KEY in your environment variables.',
      detail: 'The GOOGLE_API_KEY environment variable is required for web search functionality.'
    }, { status: 500 });
  }
  
  if (!GOOGLE_CSE_ID) {
    console.error('[Web Search] Missing GOOGLE_CSE_ID environment variable');
    return NextResponse.json({ 
      error: 'Google Custom Search Engine ID not configured. Please set GOOGLE_CSE_ID in your environment variables.',
      detail: 'The GOOGLE_CSE_ID environment variable is required for web search functionality.'
    }, { status: 500 });
  }
  
  try {
    const { query, numResults = 5 } = await req.json();
    
    if (!query || typeof query !== 'string') {
      return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
    }
    
    // Add timeout to the fetch request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
    
    const start = Date.now();
    const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_API_KEY}&cx=${GOOGLE_CSE_ID}&q=${encodeURIComponent(query)}&num=${numResults}`;
    
    const response = await fetch(searchUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    const elapsed = Date.now() - start;
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Web Search] Google API error:', response.status, errorText);
      return NextResponse.json({ 
        error: 'Web search failed', 
        detail: errorText,
        status: response.status 
      }, { status: response.status });
    }
    
    const data = await response.json() as GoogleSearchResponse;
    console.log('[Web Search] Success, elapsed:', elapsed + 'ms, results:', data.items?.length || 0);
    
    // Format the search results for the AI
    let searchResults = `Web search results for "${query}":\n\n`;
    
    if (data.items && data.items.length > 0) {
      data.items.forEach((item: GoogleSearchItem, index: number) => {
        searchResults += `${index + 1}. ${item.title}\n`;
        searchResults += `   URL: ${item.link}\n`;
        searchResults += `   Snippet: ${item.snippet}\n\n`;
      });
    } else {
      searchResults += "No relevant results found.\n";
    }
    
    return NextResponse.json({
      results: searchResults,
      raw_data: data,
      search_time: elapsed / 1000,
      query: query
    });
    
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      return NextResponse.json({ error: 'Web search timeout' }, { status: 408 });
    }
    console.error('[Web Search] Error:', err);
    return NextResponse.json({ 
      error: 'Web search failed', 
      detail: String(err) 
    }, { status: 500 });
  }
} 