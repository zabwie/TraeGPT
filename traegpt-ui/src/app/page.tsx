"use client";
import React, { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import Image from "next/image";
import { auth, signInUser, saveChatSession, loadChatSessions, deleteChatSession as fbDeleteChatSession, Message, ChatSession, uploadImageAndGetUrl, saveUserPreferences, loadUserPreferences } from './firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import { motion, AnimatePresence } from "framer-motion";

// Language detection function
function detectLanguage(code: string): string {
  const firstLine = code.trim().split('\n')[0].toLowerCase();
  
  if (firstLine.includes('import ') || firstLine.includes('from ') || firstLine.includes('def ') || firstLine.includes('class ')) return 'python';
  if (firstLine.includes('function ') || firstLine.includes('const ') || firstLine.includes('let ') || firstLine.includes('var ')) return 'javascript';
  if (firstLine.includes('public class') || firstLine.includes('private ') || firstLine.includes('public ')) return 'java';
  if (firstLine.includes('<?php') || firstLine.includes('$')) return 'php';
  if (firstLine.includes('package ') || firstLine.includes('import "') || firstLine.includes('func ')) return 'go';
  if (firstLine.includes('fn ') || firstLine.includes('let ') || firstLine.includes('struct ')) return 'rust';
  if (firstLine.includes('using ') || firstLine.includes('namespace ') || firstLine.includes('public class')) return 'csharp';
  if (firstLine.includes('html') || firstLine.includes('<!DOCTYPE')) return 'html';
  if (firstLine.includes('css') || firstLine.includes('{') && firstLine.includes(':')) return 'css';
  if (firstLine.includes('sql') || firstLine.includes('SELECT') || firstLine.includes('INSERT')) return 'sql';
  if (firstLine.includes('dockerfile') || firstLine.includes('FROM ') || firstLine.includes('RUN ')) return 'dockerfile';
  if (firstLine.includes('yaml') || firstLine.includes('---')) return 'yaml';
  if (firstLine.includes('json') || firstLine.includes('{') && firstLine.includes('"')) return 'json';
  
  return 'text';
}

// Custom CodeBlock component
function CodeBlock({ children, className }: { children: string, className?: string }) {
  const language = className?.replace('language-', '') || detectLanguage(children);
  
  const copyToClipboard = () => {
    navigator.clipboard.writeText(children);
  };
  
  return (
    <div className="code-block">
      <div className="code-header">
        <span className="code-language">{language}</span>
        <div className="code-actions">
          <button className="code-action-btn" onClick={copyToClipboard}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy
          </button>
          <button className="code-action-btn">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
            </svg>
            Edit
          </button>
        </div>
      </div>
      <div className="code-content">
        <pre style={{ margin: 0, padding: 0, background: 'none', color: 'inherit' }}>
          <code>{children}</code>
        </pre>
      </div>
    </div>
  );
}

function TypewriterMarkdown({ text }: { text: string }) {
  const [displayed, setDisplayed] = useState("");
  useEffect(() => {
    setDisplayed("");
    if (!text) return;
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(interval);
    }, 14);
    return () => clearInterval(interval);
  }, [text]);
  return (
    <div style={{ 
      whiteSpace: 'pre-wrap',
      fontFamily: 'monospace',
      lineHeight: '1.6',
      wordBreak: 'break-word'
    }}>
      <ReactMarkdown 
        components={{
          pre: ({ children }) => <>{children}</>,
          code: ({ inline, className, children }: { inline?: boolean; className?: string; children?: React.ReactNode }) => {
            return !inline ? (
              <CodeBlock className={className}>
                {String(children || '').replace(/\n$/, '')}
              </CodeBlock>
            ) : (
              <code className={className}>
                {children}
              </code>
            );
          }
        }}
      >
        {displayed}
      </ReactMarkdown>
    </div>
  );
}

// Add a helper function for timeout
function fetchWithTimeout(resource: RequestInfo, options: RequestInit = {}, timeout = 15000): Promise<Response> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Request timed out')), timeout);
    fetch(resource, options)
      .then((response) => {
        clearTimeout(timer);
        resolve(response);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [responseTime, setResponseTime] = useState<number | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [ocrLang, setOcrLang] = useState("en");
  const [categories, setCategories] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showTools, setShowTools] = useState(false);
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [fileType, setFileType] = useState<'images' | 'attachments' | 'documents'>('images');
  // Add new state for personalization
  const [userName, setUserName] = useState("");
  const [userInterests, setUserInterests] = useState("");
  const [answerStyle, setAnswerStyle] = useState("");
  const [customPersonality, setCustomPersonality] = useState("");
  const [settingsSaved, setSettingsSaved] = useState(false);
  // Add state to hold the last saved personalization settings
  const [savedUserName, setSavedUserName] = useState("");
  const [savedUserInterests, setSavedUserInterests] = useState("");
  const [savedAnswerStyle, setSavedAnswerStyle] = useState("");
  const [savedCustomPersonality, setSavedCustomPersonality] = useState("");
  // Add loading state for preferences
  const [prefsLoading, setPrefsLoading] = useState(false);
  // Add debounce ref for Firebase saves
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  // Add state to track web search
  const [isWebSearching, setIsWebSearching] = useState(false);
  // Add ref to track current chat sessions to avoid infinite loops
  const chatSessionsRef = useRef<ChatSession[]>([]);

  // Handle authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setUser(user);
      setAuthLoading(false);
      
      if (user) {
        // Load chat sessions for authenticated user
        const sessions = await loadChatSessions(user.uid);
        setChatSessions(sessions as ChatSession[]);
        chatSessionsRef.current = sessions as ChatSession[];
      }
    });

    return () => unsubscribe();
  }, []);

  // Auto-sign in if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      signInUser().catch(console.error);
    }
  }, [authLoading, user]);

  // Update current session when messages change - with debounced save
  useEffect(() => {
    if (currentSessionId && messages.length > 0 && user) {
      const updatedSession = chatSessionsRef.current.find(s => s.id === currentSessionId);
      if (updatedSession) {
        const newSession = { 
          ...updatedSession, 
          messages, 
          title: generateSessionTitle(messages),
          updatedAt: new Date() 
        };
        
        // Update local state immediately
        setChatSessions(prev => {
          const updated = prev.map(session => 
            session.id === currentSessionId ? newSession : session
          );
          chatSessionsRef.current = updated;
          return updated;
        });
        
        // Debounce Firebase save to prevent overwhelming the database
        if (saveTimeoutRef.current) {
          clearTimeout(saveTimeoutRef.current);
        }
        
        saveTimeoutRef.current = setTimeout(async () => {
          try {
            await saveChatSession(user.uid, newSession);
          } catch (error) {
            console.error('[Firebase] Error saving chat session:', error);
            // Don't show error to user for background saves
          }
        }, 2000); // Wait 2 seconds before saving to Firebase
      }
    }
    
    // Cleanup timeout on unmount
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [messages, currentSessionId, user]);

  // Load preferences on login
  useEffect(() => {
    if (user) {
      setPrefsLoading(true);
      loadUserPreferences(user.uid).then(prefs => {
        setUserName(prefs.userName || "");
        setUserInterests(prefs.userInterests || "");
        setAnswerStyle(prefs.answerStyle || "");
        setCustomPersonality(prefs.customPersonality || "");
        setSavedUserName(prefs.userName || "");
        setSavedUserInterests(prefs.userInterests || "");
        setSavedAnswerStyle(prefs.answerStyle || "");
        setSavedCustomPersonality(prefs.customPersonality || "");
        setPrefsLoading(false);
      }).catch(() => setPrefsLoading(false));
    }
  }, [user]);

  // Replace the scroll-to-bottom effect with a smart scroll that only scrolls if the user is near the bottom
  useEffect(() => {
    if (!chatContainerRef.current) return;
    const container = chatContainerRef.current;
    const threshold = 120;
    const isNearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
    if (isNearBottom) {
      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
      });
    }
  }, [messages, loading]);

  // Auto-resize textarea
  React.useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  function generateSessionTitle(messages: Message[]): string {
    if (messages.length === 0) return "New chat";
    
    const firstUserMessage = messages.find(msg => msg.role === "user");
    if (firstUserMessage) {
      const content = firstUserMessage.content;
      return content.length > 30 ? content.substring(0, 30) + "..." : content;
    }
    
    return "New chat";
  }

  function handleNewChat() {
    setMessages([]);
    setInput("");
    setImage(null);
    setImagePreviewUrl(null);
    setError(null);
    setShowTools(false);
    setShowPlusMenu(false);
    setCurrentSessionId(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function loadChatSession(sessionId: string) {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setMessages(session.messages);
      setCurrentSessionId(sessionId);
      setInput("");
      setImage(null);
      setImagePreviewUrl(null);
      setError(null);
      setShowTools(false);
      setShowPlusMenu(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDeleteChatSession(sessionId: string) {
    if (!user) return;
    
    try {
      await fbDeleteChatSession(user.uid, sessionId);
      setChatSessions(prev => {
        const updated = prev.filter(s => s.id !== sessionId);
        chatSessionsRef.current = updated;
        return updated;
      });
      if (currentSessionId === sessionId) {
        handleNewChat();
      }
    } catch (error) {
      console.error('Error deleting session:', error);
      setError("Failed to delete chat session.");
    }
  }

  async function handleSaveSettings() {
    if (!user) return;
    setSettingsSaved(true);
    setSavedUserName(userName);
    setSavedUserInterests(userInterests);
    setSavedAnswerStyle(answerStyle);
    setSavedCustomPersonality(customPersonality);
    try {
      await saveUserPreferences(user.uid, {
        userName,
        userInterests,
        answerStyle,
        customPersonality
      });
      setTimeout(() => setSettingsSaved(false), 1500);
    } catch (error) {
      console.error('[Firebase] Error saving preferences:', error);
      setError("Failed to save preferences. Please try again.");
      setTimeout(() => setSettingsSaved(false), 1500);
    }
  }

  function buildSystemPrompt() {
    let prompt = "You are Trae, an empathetic AI assistant created by Zabi. Respond as Trae, never as other models.";
    
    prompt += "\n\nCRITICAL FORMATTING REQUIREMENTS: You MUST format ALL responses beautifully with proper structure and indentation. The browser now preserves whitespace perfectly, so use this to your advantage:";
    prompt += "\n• **Bold text** for important terms, names, and key information";
    prompt += "\n• Bullet points (•) for lists and features";
    prompt += "\n• Numbered lists (1., 2., 3.) for steps or sequential info";
    prompt += "\n• **HEADERS**: Use # ## ### for different font sizes and hierarchy";
    prompt += "\n• **PERFECT SPACING**: Use blank lines between sections to create visual breathing room";
    prompt += "\n• **ADVANCED INDENTATION**: Use multiple levels with arrows and hanging indents";
    prompt += "\n    → First level: 4 spaces";
    prompt += "\n        → Second level: 8 spaces";
    prompt += "\n            → Third level: 12 spaces";
    prompt += "\n                → Fourth level: 16 spaces";
    prompt += "\n• **HANGING INDENTS**: When text wraps, align with the start of the first line";
    prompt += "\n• **ARROW INDICATORS**: Use → for sub-points and variations";
    prompt += "\n• **MANDATORY SPACING**: Always add blank lines between sections to prevent compression";
    prompt += "\n• **SECTION BREAKS**: Use double line breaks between major sections";
    prompt += "\n• **VISUAL HIERARCHY**: Use bigger headings and more indentation to make text stand out";
    prompt += "\n• **CODE INTEGRATION**: Include code snippets with proper syntax highlighting";
    prompt += "\n• **METADATA**: Add social media metrics and technical details";
    prompt += "\n• **MONOSPACE FRIENDLY**: Since we use monospace font, align things perfectly";
    prompt += "\n• **FONT SIZES**: Use # for main titles, ## for sections, ### for subsections";
    prompt += "\n• **EMPHASIS**: Use ***italic bold*** for extra emphasis";
    prompt += "\n• **QUOTES**: Use > for important callouts and tips";
    prompt += "\n• **CODE BLOCKS**: Always use proper code blocks with language detection";
    prompt += "\n• **PRECISE INDENTATION**: Code languages are strict with indentation - be exact";
    prompt += "\n• **LANGUAGE DETECTION**: The system will automatically detect Python, JavaScript, CSS, HTML, SQL, etc.";
    prompt += "\n• **POEM FORMATTING**: For poems, use clean minimalist formatting:";
    prompt += "\n    → Start with a simple introduction line";
    prompt += "\n    → Add a thin separator line (---)";
    prompt += "\n    → Use **bold** for the title";
    prompt += "\n    → Separate stanzas with empty lines (double line breaks)";
    prompt += "\n    → Keep lines short and left-aligned";
    prompt += "\n    → No excessive formatting, just clean text";
    prompt += "\n    → NO ARROWS (→) in poems - keep it clean and simple";
    
    prompt += "\n\nEXAMPLE FORMAT WITH PERFECT SPACING:";
    prompt += "\n# **3 Essential Tools for Your Coding Life**";
    prompt += "\n\n## **Quick Setup Tools**";
    prompt += "\n\n• **VS Code** – Your daily driver for **code editing**";
    prompt += "\n    → Extensions: **Prettier**, **GitLens**, **Live Share**";
    prompt += "\n    → Pro tip: Enable **Zen Mode** for deep focus";
    prompt += "\n\n• **GitHub Desktop** – When CLI feels too heavy";
    prompt += "\n    → Drag-and-drop commits";
    prompt += "\n    → Visual diff viewer";
    prompt += "\n    → Push without touching terminal";
    prompt += "\n\n• **Postman** – **API testing** made simple";
    prompt += "\n    → Save requests as collections";
    prompt += "\n    → Test auth flows";
    prompt += "\n    → Share with your team";
    prompt += "\n\n## **Use them in this order:**";
    prompt += "\n\n    **VS Code** for writing";
    prompt += "\n    **GitHub Desktop** for versioning";
    prompt += "\n    **Postman** for testing";
    prompt += "\n\n> **Pro tip:** Takes 10 minutes to set up all three.";
    prompt += "\n\n**Example Python code with precise indentation:**";
    prompt += "\n```python";
    prompt += "\nimport requests";
    prompt += "\n";
    prompt += "\nAPI_KEY = 'your_api_key_here'";
    prompt += "\nCX = 'your_search_engine_id_here'";
    prompt += "\nquery = 'test search'";
    prompt += "\nurl = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}'";
    prompt += "\n";
    prompt += "\nresponse = requests.get(url)";
    prompt += "\nprint(response.json())";
    prompt += "\n```";
    prompt += "\n\n**Example poem with clean formatting:**";
    prompt += "\n**Moonlight Sonata**";
    prompt += "\n---";
    prompt += "\n\nA silver thread of moonlight";
    prompt += "\nslips through the open window";
    prompt += "\npooling like liquid memory";
    prompt += "\nacross the wooden floor";
    prompt += "\n\nIt carries the scent of **rain-soaked earth**";
    prompt += "\nand the distant echo";
    prompt += "\nof laughter from another life";
    prompt += "\nwhen we were infinite";
    prompt += "\n\nThe clock ticks its **ancient rhythm**";
    prompt += "\nmarking time in heartbeats";
    prompt += "\nwhile shadows dance";
    prompt += "\nwith the grace of forgotten dreams";
    prompt += "\n\nIn this **quiet cathedral of night**";
    prompt += "\nwhere every breath is a prayer";
    prompt += "\nand every thought a confession";
    prompt += "\nI find you still";
    
    prompt += "\n\nIMPORTANT: You have access to web search capabilities. When users ask about current events, recent information, or anything that might require up-to-date data, you should automatically search the web to provide accurate information. Do not mention that you're searching - just do it seamlessly and provide the information naturally.";
    
    prompt += "\n\nTo search the web, include this special format in your response: [WEB_SEARCH:query]. For example: [WEB_SEARCH:latest iPhone release date] or [WEB_SEARCH:current weather in New York]. The search will happen automatically and you'll receive the results to respond with.";
    
    if (savedUserName) prompt += ` User's name: ${savedUserName}.`;
    if (savedUserInterests) prompt += ` Interests: ${savedUserInterests}.`;
    if (savedAnswerStyle) {
      const styleMap = {
        'friendly': 'Be conversational and warm.',
        'formal': 'Be professional and formal.',
        'concise': 'Be brief and to the point.',
        'detailed': 'Provide thorough explanations.'
      };
      prompt += ` ${styleMap[savedAnswerStyle as keyof typeof styleMap] || ''}`;
    }
    if (savedCustomPersonality) prompt += ` ${savedCustomPersonality}`;
    
    return prompt;
  }

  async function sendMessage() {
    console.log('[sendMessage] Function called');
    try {
      if (!input.trim() && !image) {
        console.log('[sendMessage] No input or image, returning early');
        return;
      }
      if (!user) {
        console.log('[sendMessage] No user, returning early');
        return;
      }
      
      console.log('[sendMessage] Starting message processing...');
      setLoading(true);
      setError(null);
      
      let newMessages = [...messages];
      let hasTextInput = false;
      console.log('[sendMessage] Input trim result:', input.trim());
      if (input.trim()) {
        hasTextInput = true;
        console.log('[sendMessage] Processing text input:', input.substring(0, 50) + '...');
        const userMessage = { role: "user" as const, content: input };
        console.log('[sendMessage] Created user message:', userMessage);
        newMessages.push(userMessage);
        setMessages(newMessages);
        
        setInput("");
        
        // Training data saving removed - Kimi is already well-trained
        console.log('[sendMessage] Message processed, proceeding to AI...');
      }
      
      if (image) {
        console.log('[sendMessage] Processing image...');
        try {
          // 1. Upload image to Firebase Storage and get persistent URL
          const imageUrl = await uploadImageAndGetUrl(image, user.uid);
          // 2. Add the message with the persistent imageUrl
          newMessages = [
            ...newMessages,
            {
              role: "user",
              content: "[Image uploaded]",
              imageUrl,
            },
          ];
          setMessages(newMessages);
          
          // 3. Send image to HuggingFace OWL-ViT for intelligent analysis
          const form = new FormData();
          form.append("file", image);
          try {
            console.log('[sendMessage] Sending image to HuggingFace OWL-ViT...');
            const analyzeRes = await fetchWithTimeout("/api/image/analyze", { 
              method: "POST", 
              body: form 
            }, 60000); // 60 seconds timeout for image analysis
            
            console.log('[sendMessage] Image analysis response status:', analyzeRes.status);
            
            if (!analyzeRes.ok) {
              const errorText = await analyzeRes.text();
              console.error('[sendMessage] Image analysis failed with status:', analyzeRes.status, 'error:', errorText);
              throw new Error(`Image analysis failed: ${analyzeRes.status} - ${errorText}`);
            }
            
            const analyzeData = await analyzeRes.json();
            console.log('[sendMessage] Image analysis result:', analyzeData);
            
            // Create a user message with the image description
            const imageDescription = analyzeData.description || "I can see various objects in this image.";
            const userImageMessage = { 
              role: "user" as const, 
              content: imageDescription 
            };
            
            // Add the image description as a user message
            newMessages = [...newMessages, userImageMessage];
            setMessages(newMessages);
            
            console.log('[sendMessage] Added image description to conversation:', imageDescription);
            
          } catch (e) {
            console.error('[sendMessage] Image analysis error:', e);
            const errorMessage = e instanceof Error ? e.message : 'Image analysis failed or timed out. Please try again.';
            setError(errorMessage);
            setLoading(false);
            return;
          }
          
          setImage(null);
          setImagePreviewUrl(null);
          if (fileInputRef.current) fileInputRef.current.value = "";
        } catch (e) {
          console.error('Image upload error:', e);
          setError(`Image upload failed: ${e instanceof Error ? e.message : 'Unknown error'}`);
          setLoading(false);
          return;
        }
      }
      
      // Send to TogetherAI if we have text input OR if we just processed an image
      const shouldSendToAI = hasTextInput || image;
      console.log('[sendMessage] Should send to TogetherAI:', shouldSendToAI, 'hasTextInput:', hasTextInput, 'image:', !!image);
      
      if (shouldSendToAI) {
        console.log('[sendMessage] Sending to TogetherAI...');
        try {
          console.log('[Chat] Sending message to TogetherAI...');
          const start = Date.now();
          const res = await fetchWithTimeout("/api/togetherai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              messages: [
                { role: 'system', content: buildSystemPrompt() },
                ...newMessages.map((m) => ({ role: m.role, content: m.content })),
              ],
            }),
          }, 60000); // 60 seconds timeout
          const elapsed = Date.now() - start;
          setResponseTime(elapsed);
          console.log(`[Chat] TogetherAI response time: ${elapsed}ms`);
          if (!res.ok) {
            console.error(`[Chat] TogetherAI error: ${res.status} ${res.statusText}`);
            throw new Error(`Chat failed: ${res.status} ${res.statusText}`);
          }
          const data = await res.json();
          console.log('[Chat] TogetherAI response data:', data);
          const assistantMessage: Message = { 
            role: "assistant" as const, 
            content: data.choices?.[0]?.message?.content || "[No response]" 
          };
          
          // Check if the AI wants to search the web
          const webSearchMatch = assistantMessage.content.match(/\[WEB_SEARCH:(.*?)\]/);
          
          if (webSearchMatch) {
            const searchQuery = webSearchMatch[1].trim();
            console.log('[Web Search] AI requested search for:', searchQuery);
            setIsWebSearching(true); // Set web search flag
            try {
              // Perform the web search
              const searchRes = await fetchWithTimeout("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: searchQuery, numResults: 5 }),
              }, 60000);
              
              if (!searchRes.ok) {
                throw new Error("Web search failed");
              }
              
              const searchData = await searchRes.json();
              console.log('[Web Search] Search completed, results:', searchData.results.substring(0, 100) + '...');
              
              // Add the search results as a system message
              const searchResultsMessage: Message = {
                role: "system" as const,
                content: searchData.results
              };
              
              // Update messages with both the AI response and search results
              const updatedMessages = [...newMessages, assistantMessage, searchResultsMessage];
              setMessages(updatedMessages);
              
              // Add a delay to prevent rate limiting
              await new Promise(resolve => setTimeout(resolve, 1000));
              
              // Now get the AI's final response with the search results
              console.log('[Chat] Getting AI response with search results...');
              const finalRes = await fetchWithTimeout("/api/togetherai", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  messages: [
                    { role: 'system', content: buildSystemPrompt() },
                    ...updatedMessages.map((m) => ({ role: m.role, content: m.content })),
                  ],
                }),
              }, 60000);
              
              if (!finalRes.ok) {
                if (finalRes.status === 429) {
                  throw new Error("Rate limit exceeded. Please wait a moment and try again.");
                }
                throw new Error("Final AI response failed");
              }
              
              const finalData = await finalRes.json();
              const finalAssistantMessage: Message = {
                role: "assistant" as const,
                content: finalData.choices?.[0]?.message?.content || "[No response]"
              };
              
              // Replace the messages with the final response
              const finalMessages = [...newMessages, finalAssistantMessage];
              setMessages(finalMessages);
              
              // Update chat session with final messages
              if (!currentSessionId) {
                const newSession: ChatSession = {
                  id: Date.now().toString(),
                  title: generateSessionTitle(finalMessages),
                  messages: finalMessages,
                  createdAt: new Date(),
                  updatedAt: new Date()
                };
                setChatSessions(prev => {
                  const updated = [newSession, ...prev];
                  chatSessionsRef.current = updated;
                  return updated;
                });
                setCurrentSessionId(newSession.id);
                try {
                  await saveChatSession(user.uid, newSession);
                } catch (error) {
                  console.error('[sendMessage] Error saving new chat session:', error);
                }
              } else {
                const updatedSession = {
                  ...chatSessionsRef.current.find(s => s.id === currentSessionId)!,
                  messages: finalMessages,
                  title: generateSessionTitle(finalMessages),
                  updatedAt: new Date()
                };
                setChatSessions(prev => {
                  const updated = prev.map(session => 
                    session.id === currentSessionId ? updatedSession : session
                  );
                  chatSessionsRef.current = updated;
                  return updated;
                });
              }
              
            } catch (searchError) {
              console.error('[Web Search] Error:', searchError);
              // If search fails, just show the original AI response
              setMessages((msgs) => [...msgs, assistantMessage]);
              if (searchError instanceof Error) {
                setError(searchError.message);
              }
            } finally {
              setIsWebSearching(false); // Reset web search flag
            }
          } else {
            // No web search needed, just add the AI response
            setMessages((msgs) => [...msgs, assistantMessage]);
          }
          
          // Create or update chat session - with debounced save
          const updatedMessages: Message[] = [...newMessages, assistantMessage];
          if (!currentSessionId) {
            // Create new session
            const newSession: ChatSession = {
              id: Date.now().toString(),
              title: generateSessionTitle(updatedMessages),
              messages: updatedMessages,
              createdAt: new Date(),
              updatedAt: new Date()
            };
            setChatSessions(prev => {
              const updated = [newSession, ...prev];
              chatSessionsRef.current = updated;
              return updated;
            });
            setCurrentSessionId(newSession.id);
            // Save to Firebase with error handling
            try {
              await saveChatSession(user.uid, newSession);
            } catch (error) {
              console.error('[sendMessage] Error saving new chat session:', error);
              // Don't show error to user for background saves
            }
          } else {
            // Update existing session - let the useEffect handle the save
            const updatedSession = {
              ...chatSessionsRef.current.find(s => s.id === currentSessionId)!,
              messages: updatedMessages, 
              title: generateSessionTitle(updatedMessages),
              updatedAt: new Date() 
            };
            setChatSessions(prev => {
              const updated = prev.map(session => 
                session.id === currentSessionId ? updatedSession : session
              );
              chatSessionsRef.current = updated;
              return updated;
            });
            // The useEffect will handle the debounced save
          }
          console.log('[Chat] Message handling complete.');
        } catch (e) {
          if (e instanceof Error && e.message === 'Request timed out') {
            console.error('[Chat] TogetherAI request timed out.');
            setError("AI response timed out. Please try again.");
          } else {
            console.error('[Chat] Error:', e);
            setError(`Chat failed: ${e instanceof Error ? e.message : 'Unknown error'}.`);
          }
        }
      } else {
        console.log('[sendMessage] No text input and no image, not sending to TogetherAI');
      }
      console.log('[sendMessage] Function ending, setting loading to false');
      setLoading(false);
      // Clear response time after a few seconds
      setTimeout(() => setResponseTime(null), 5000);
    } catch (error) {
      console.error('[sendMessage] Unexpected error:', error);
      setError(`Unexpected error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setLoading(false);
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];
    setShowPlusMenu(false);
    if (fileType === 'images') {
      setImage(file);
      setImagePreviewUrl(URL.createObjectURL(file));
    } else {
      // Upload file to Firebase Storage and add as message
      uploadImageAndGetUrl(file, user?.uid || 'unknown').then((url) => {
        setMessages((msgs) => [
          ...msgs,
          {
            role: 'user',
            content: fileType === 'documents' ? '[Document uploaded]' : '[Attachment uploaded]',
            fileUrl: url,
            fileName: file.name,
            fileType: file.type,
          },
        ]);
      });
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function renderMessage(msg: Message, i: number, isLatestAssistant: boolean) {
    // Only show image preview for the current image before upload/analysis
    if (i === messages.length && imagePreviewUrl) {
      return (
        <div key={i} className="message-row user">
          <div className="chat-bubble">
            <Image src={imagePreviewUrl} alt="preview" width={320} height={240} className="rounded-lg shadow-sm" />
          </div>
        </div>
      );
    }
    if (msg.imageUrl) {
      return (
        <div key={i} className="message-row user">
          <div className="chat-bubble">
            <Image src={msg.imageUrl} alt="uploaded" width={320} height={240} className="rounded-lg shadow-sm" />
          </div>
        </div>
      );
    }
    if (msg.imageResult) {
      const result = msg.imageResult;
      return (
        <div key={i} className="flex justify-start mb-4">
          <div style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }} className="max-w-3xl rounded-2xl shadow-sm border border-gray-700 p-4">
            <h3 className="text-lg font-medium text-gray-100 mb-3">Image Analysis Results</h3>
            
            {/* Classification */}
            {result.classification && result.classification.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Classification:</h4>
                <div className="flex flex-wrap gap-2">
                  {result.classification.map((item: { class: string; confidence: number }, idx: number) => (
                    <span key={idx} style={{ background: 'var(--success-bg)', color: 'var(--success-text)', borderRadius: '0.375rem', padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
                      {item.class}: {(item.confidence * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {/* Object Detection */}
            {result.object_detection && result.object_detection.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Objects Detected:</h4>
                <div className="flex flex-wrap gap-2">
                  {result.object_detection.map((obj: { class: string; confidence: number }, idx: number) => (
                    <span key={idx} style={{ background: 'var(--success-bg)', color: 'var(--success-text)', borderRadius: '0.375rem', padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
                      {obj.class}: {(obj.confidence * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {/* Caption */}
            {result.caption && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Caption:</h4>
                <div style={{ background: 'var(--warning-bg)', border: '1px solid var(--warning-border)', borderRadius: '0.5rem', padding: '0.75rem' }}>
                  <p className="text-sm text-gray-200">{result.caption}</p>
                  {result.caption.includes('[Note:') && (
                    <p className="text-xs text-yellow-400 mt-1">⚠️ AI may have added details not visible in the image</p>
                  )}
                </div>
              </div>
            )}
            
            {/* Text Extraction */}
            {result.text_extraction && result.text_extraction.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Text Found:</h4>
                <div style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', borderRadius: '0.5rem' }}>
                  {result.text_extraction.map((text: { text: string; confidence: number }, idx: number) => (
                    <p key={idx} className="text-sm text-gray-200">
                      &quot;{text.text}&quot; ({(text.confidence * 100).toFixed(1)}% confidence)
                    </p>
                  ))}
                </div>
              </div>
            )}
            
            {/* Analysis Time */}
            {result.analysis_time && (
              <div className="text-xs text-gray-500 mt-3">
                Analysis completed in {result.analysis_time.toFixed(2)} seconds
              </div>
            )}
          </div>
        </div>
      );
    }
    if (msg.fileUrl) {
      return (
        <div key={i} className={`message-row ${msg.role === "user" ? "user" : "assistant"}`}>
          <div className="chat-bubble">
            <a href={msg.fileUrl} target="_blank" rel="noopener noreferrer" className="file-link">
              {msg.fileName || 'Download file'}
            </a>
          </div>
        </div>
      );
    }
    // Typewriter only for latest assistant message
    const isAssistant = msg.role === "assistant";
    return (
      <div key={i} className={`message-row ${msg.role === "user" ? "user" : "assistant"}`}>
        <div className="chat-bubble">
          <div className="prose prose-sm max-w-none prose-invert" style={{ 
            whiteSpace: 'pre-wrap',
            fontFamily: 'monospace',
            lineHeight: '1.6',
            wordBreak: 'break-word'
          }}>
            {isAssistant && isLatestAssistant ? (
              <TypewriterMarkdown text={msg.content} />
            ) : (
              <ReactMarkdown 
                components={{
                  pre: ({ children }) => <>{children}</>,
                  code: ({ inline, className, children }: { inline?: boolean; className?: string; children?: React.ReactNode }) => {
                    return !inline ? (
                      <CodeBlock className={className}>
                        {String(children || '').replace(/\n$/, '')}
                      </CodeBlock>
                    ) : (
                      <code className={className}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {msg.content}
              </ReactMarkdown>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--chat-bg)' }}>
      {/* Left Sidebar - Fixed */}
      {sidebarOpen && (
        <div
          className="w-64 sidebar"
          style={{
            position: 'fixed',
            left: 0,
            top: 0,
            height: '100vh',
            background: 'var(--sidebar-bg)',
            borderRight: '1px solid var(--sidebar-border)',
            color: 'var(--text-main)',
            zIndex: 10
          }}
        >
          <div className="p-4" style={{ borderBottom: '1px solid var(--sidebar-border)' }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2" style={{ color: 'var(--text-main)' }}>
                <div style={{ background: 'var(--sidebar-icon-bg)', color: 'var(--sidebar-icon-color)', width: 32, height: 32, borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <span className="text-lg font-semibold">TraeGPT</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>
              <button onClick={() => setSidebarOpen(false)} style={{ color: 'var(--text-secondary)' }} className="p-1 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
          
          <div className="flex-1 p-4 space-y-1 overflow-y-auto">
            <button onClick={handleNewChat} style={{ color: 'var(--text-secondary)', background: 'none' }} className="w-full p-3 rounded-lg flex items-center gap-3 transition-colors sidebar-newchat">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <span className="text-sm">New chat</span>
            </button>
            
            {/* Chat History */}
            <div className="mt-4 space-y-1">
              {chatSessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => loadChatSession(session.id)}
                  style={{ background: currentSessionId === session.id ? 'var(--sidebar-selected-bg)' : 'none', color: currentSessionId === session.id ? 'var(--text-main)' : 'var(--text-secondary)' }}
                  className={`w-full p-3 rounded-lg flex items-center justify-between group cursor-pointer transition-colors sidebar-session${currentSessionId === session.id ? ' selected' : ''}`}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <span className="text-sm truncate">{session.title}</span>
                  </div>
                  <button
                    onClick={() => handleDeleteChatSession(session.id)}
                    style={{ color: 'var(--text-secondary)' }}
                    className="opacity-0 group-hover:opacity-100 p-1 transition-all"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Content - Adjust margin when sidebar is open */}
      <div className={`flex-1 flex flex-col ${sidebarOpen ? 'ml-64' : ''}`}>
        {/* Top Bar - Fixed */}
        <div style={{ background: 'var(--chat-bg)', borderBottom: '1px solid var(--sidebar-border)', color: 'var(--text-main)' }} className="px-6 py-3 flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-4">
            {!sidebarOpen && (
              <button onClick={() => setSidebarOpen(true)} style={{ color: 'var(--text-secondary)' }} className="p-2 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}
            <h1 className="text-lg font-semibold" style={{ color: 'var(--text-main)' }}>TraeGPT</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button onClick={() => setShowTools(!showTools)} style={{ color: 'var(--text-secondary)' }} className="flex items-center gap-1 transition-colors">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span className="text-sm">Tools</span>
            </button>
            {!user && (
              <button
                onClick={signInUser}
                style={{ background: 'var(--button-bg)', color: 'var(--text-main)', border: 'none', borderRadius: 'var(--radius)', padding: '8px 18px', fontWeight: 500, fontSize: '1rem', cursor: 'pointer', transition: 'background 0.2s' }}
              >
                Sign up / Log in
              </button>
            )}
          </div>
        </div>

        {/* Chat Container - Add top padding to account for fixed header */}
        <div className="flex-1 flex flex-col pt-0">
          <div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto px-6 py-6 space-y-4 pb-0"
          >
            {messages.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '60vh', width: '100%' }}>
                <p style={{ color: 'var(--text-main)', fontSize: '2rem', fontWeight: 500, textAlign: 'center', margin: 0 }}>
                  How can I help?
                </p>
              </div>
            )}
            {messages.map((msg, i) => renderMessage(msg, i, i === messages.length - 1 && msg.role === "assistant"))}
            {loading && (
              <div className="flex justify-start mb-4">
                <div style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-secondary)' }} className="rounded-2xl shadow-sm px-4 py-3">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div style={{ background: 'var(--text-secondary)' }} className="w-2 h-2 rounded-full animate-bounce"></div>
                      <div style={{ background: 'var(--text-secondary)', animationDelay: '0.1s' }} className="w-2 h-2 rounded-full animate-bounce"></div>
                      <div style={{ background: 'var(--text-secondary)', animationDelay: '0.2s' }} className="w-2 h-2 rounded-full animate-bounce"></div>
                    </div>
                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {isWebSearching ? "Searching the web..." : "AI is thinking..."}
                      {responseTime && <span style={{ color: 'var(--text-success)' }}> ({responseTime}ms)</span>}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

          {/* Input Area - Fixed to Bottom */}
          <div style={{ background: 'var(--chat-bg)', borderTop: '1px solid var(--sidebar-border)' }} className="border-t border-gray-700 bg-gray-900 p-6 sticky bottom-0">
            <div className="max-w-4xl mx-auto">
              {/* Tools Panel */}
              {showTools && (
                <div style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }} className="mb-4 p-4 rounded-lg">
                  {prefsLoading && (
                    <div style={{ color: 'var(--text-secondary)', marginBottom: '1rem', textAlign: 'center', fontWeight: 500 }}>
                      Loading preferences...
                    </div>
                  )}
                  {!user && (
                    <div style={{ color: 'var(--text-warning)', marginBottom: '1rem', textAlign: 'center', fontWeight: 500 }}>
                      Please log in to save your preferences.
                    </div>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Your Name</label>
                      <input
                        type="text"
                        value={userName}
                        onChange={e => setUserName(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg"
                        style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }}
                        placeholder="e.g. Zabi"
                        disabled={loading || !user || prefsLoading}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Your Interests</label>
                      <input
                        type="text"
                        value={userInterests}
                        onChange={e => setUserInterests(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg"
                        style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }}
                        placeholder="e.g. coding, music, AI"
                        disabled={loading || !user || prefsLoading}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Preferred Answer Style</label>
                      <select
                        value={answerStyle}
                        onChange={e => setAnswerStyle(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg"
                        style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }}
                        disabled={loading || !user || prefsLoading}
                      >
                        <option value="">No preference</option>
                        <option value="friendly">Friendly</option>
                        <option value="formal">Formal</option>
                        <option value="concise">Concise</option>
                        <option value="detailed">Detailed</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Custom AI Personality Prompt</label>
                      <textarea
                        value={customPersonality}
                        onChange={e => setCustomPersonality(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg"
                        style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)', minHeight: '60px' }}
                        placeholder="e.g. Always answer like a pirate!"
                        disabled={loading || !user || prefsLoading}
                      />
                    </div>
                    {/* Existing OCR and CLIP fields below */}
                    <div>
                      <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>OCR Languages</label>
                      <input
                        type="text"
                        value={ocrLang}
                        onChange={e => setOcrLang(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }}
                        placeholder="en,es,fr"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>CLIP Categories</label>
                      <input
                        type="text"
                        value={categories}
                        onChange={e => setCategories(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)', color: 'var(--text-main)' }}
                        placeholder="cat,dog,car"
                        disabled={loading}
                      />
                    </div>
                  </div>
                  <div className="col-span-full flex items-center gap-3 mt-4">
                    <button
                      onClick={handleSaveSettings}
                      style={{ background: 'var(--button-bg)', color: 'var(--text-main)', border: 'none', borderRadius: 'var(--radius)', padding: '10px 24px', cursor: !user ? 'not-allowed' : 'pointer', fontWeight: 500, fontSize: '1rem', transition: 'background 0.2s' }}
                      disabled={loading || !user || prefsLoading}
                    >
                      Save
                    </button>
                    {settingsSaved && <span style={{ color: 'var(--text-success)', fontSize: '0.95rem' }}>Saved!</span>}
                  </div>
                </div>
              )}

              {error && (
                <div style={{ color: 'var(--text-warning)' }} className="text-sm mb-3 text-center">{error}</div>
              )}

              <div style={{ background: 'var(--input-bar-bg)', border: '1px solid var(--input-bar-border)' }} className="flex items-end gap-3 rounded-2xl px-4 py-3">
                <div className="relative">
                  <button 
                    onClick={() => setShowPlusMenu(!showPlusMenu)}
                    className="p-2 text-gray-400 hover:text-gray-300 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </button>
                  
                  {showPlusMenu && (
                    <AnimatePresence>
                      <motion.div
                        initial={{ opacity: 0, scale: 0.8, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.8, y: 20 }}
                        transition={{ type: "spring", stiffness: 300, damping: 20 }}
                        className="absolute bottom-full left-0 mb-2 bg-gray-800/90 backdrop-blur-lg rounded-xl shadow-2xl border border-gray-700 p-2 min-w-48 z-50"
                      >
                        <div className="space-y-1">
                          <button 
                            onClick={() => {
                              setFileType('images');
                              if (fileInputRef.current) {
                                fileInputRef.current.accept = 'image/*';
                                fileInputRef.current.click();
                              }
                              setShowPlusMenu(false);
                            }}
                            className="w-full p-2 text-gray-300 hover:text-white hover:bg-gray-600 rounded flex items-center gap-2 text-sm"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            Images
                          </button>
                          <button 
                            onClick={() => {
                              setFileType('attachments');
                              if (fileInputRef.current) {
                                fileInputRef.current.accept = '';
                                fileInputRef.current.click();
                              }
                              setShowPlusMenu(false);
                            }}
                            className="w-full p-2 text-gray-300 hover:text-white hover:bg-gray-600 rounded flex items-center gap-2 text-sm"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                            </svg>
                            Attachments
                          </button>
                          <button 
                            onClick={() => {
                              setFileType('documents');
                              if (fileInputRef.current) {
                                fileInputRef.current.accept = '.pdf,.doc,.docx,.txt,.ppt,.pptx,.xls,.xlsx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document';
                                fileInputRef.current.click();
                              }
                              setShowPlusMenu(false);
                            }}
                            className="w-full p-2 text-gray-300 hover:text-white hover:bg-gray-600 rounded flex items-center gap-2 text-sm"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Documents
                          </button>
                        </div>
                      </motion.div>
                    </AnimatePresence>
                  )}
                </div>
                
                <div className="flex-1">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Message TraeGPT..."
                    className="w-full resize-none bg-transparent border-none outline-none text-white placeholder-gray-400 text-base text-center"
                    rows={1}
                    disabled={loading}
                    style={{ minHeight: '44px', maxHeight: '200px' }}
                  />
                </div>
                
                <div className="flex items-center gap-2">
                  <button className="p-2 text-gray-400 hover:text-gray-300 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </button>
                  
                  <button
                    onClick={() => {
                      console.log('[Send Button] Clicked');
                      sendMessage();
                    }}
                    disabled={loading || (!input.trim() && !image)}
                    className="p-2 bg-gray-600 text-white rounded-lg hover:bg-gray-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    title="Send message"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </div>
              </div>
                        {/* After the input bar's closing div, add the disclaimer below it */}
                <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.95rem', margin: '8px 0 0 0', opacity: 0.8 }}>
                    Note: For some reason it&apos;s a bit sensetive so if you send a message immediately after the AI, it will show an error. To prevent this, please wait 3-5 seconds.
                </div>
              {imagePreviewUrl && (
                <div className="mt-3 flex items-center gap-2 text-sm text-gray-400">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span>Image ready to upload</span>
                  <Image src={imagePreviewUrl} alt="preview" width={80} height={60} className="rounded ml-2" />
                </div>
              )}
            </div>
          </div>
        </div>
      
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
