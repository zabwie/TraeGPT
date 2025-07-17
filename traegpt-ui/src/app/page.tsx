"use client";
import React, { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import Image from "next/image";
import { auth, signInUser, saveChatSession, loadChatSessions, deleteChatSession as fbDeleteChatSession, saveTrainingData, Message, ChatSession, uploadImageAndGetUrl } from './firebase';
import { onAuthStateChanged, User } from 'firebase/auth';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
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

  // Handle authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setUser(user);
      setAuthLoading(false);
      
      if (user) {
        // Load chat sessions for authenticated user
        const sessions = await loadChatSessions(user.uid);
        setChatSessions(sessions as ChatSession[]);
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

  // Update current session when messages change
  useEffect(() => {
    if (currentSessionId && messages.length > 0 && user) {
      const updatedSession = chatSessions.find(s => s.id === currentSessionId);
      if (updatedSession) {
        const newSession = { 
          ...updatedSession, 
          messages, 
          title: generateSessionTitle(messages),
          updatedAt: new Date() 
        };
        
        // Update local state
        setChatSessions(prev => prev.map(session => 
          session.id === currentSessionId ? newSession : session
        ));
        
        // Save to Firebase
        saveChatSession(user.uid, newSession).catch(console.error);
      }
    }
  }, [messages, currentSessionId, user]);

  React.useEffect(() => {
    // Scroll to bottom on new message
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
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

  async function handleDeleteChatSession(sessionId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!user) return;
    
    try {
      await fbDeleteChatSession(user.uid, sessionId);
      setChatSessions(prev => prev.filter(s => s.id !== sessionId));
      if (currentSessionId === sessionId) {
        handleNewChat();
      }
    } catch (error) {
      console.error('Error deleting session:', error);
      setError("Failed to delete chat session.");
    }
  }

  async function sendMessage() {
    if (!input.trim() && !image) return;
    if (!user) return;
    
    setLoading(true);
    setError(null);
    
    let newMessages = [...messages];
    if (input.trim()) {
      const userMessage = { role: "user" as const, content: input };
      newMessages.push(userMessage);
      setMessages(newMessages);
      setInput("");
      
      // Save user message for training
      await saveTrainingData(user.uid, {
        role: "user",
        content: input
      });
    }
    
    if (image) {
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
        // 3. Send image to backend for analysis
        const form = new FormData();
        form.append("file", image);
        form.append("languages", ocrLang);
        if (categories) form.append("categories", categories);
        form.append("analysis_type", "full");
        try {
          const res = await fetch(`${API_BASE}/v1/image/analyze`, {
            method: "POST",
            body: form,
          });
          if (!res.ok) {
            throw new Error(`Image analysis failed: ${res.status} ${res.statusText}`);
          }
          const result = await res.json();
          // Compose a summary string for the AI
          let summary = "Image analysis:";
          if (result.classification && result.classification.length > 0) {
            summary += ` Classification: ${result.classification.map((c: { class: string; confidence: number }) => c.class).join(", ")}.`;
          }
          if (result.object_detection && result.object_detection.length > 0) {
            summary += ` Objects detected: ${result.object_detection.map((o: { class: string; confidence: number }) => o.class).join(", ")}.`;
          }
          if (result.caption) {
            summary += ` Caption: ${result.caption}`;
          }
          if (result.text_extraction && result.text_extraction.length > 0) {
            summary += ` Text found: ${result.text_extraction.map((t: { text: string; confidence: number }) => t.text).join(", ")}.`;
          }
          setMessages((msgs) => [
            ...msgs,
            {
              role: "assistant",
              content: summary,
              imageResult: result,
            },
          ]);
        } catch (e) {
          setError("Image analysis failed.");
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
    
    if (input.trim()) {
      try {
        const res = await fetch(`${API_BASE}/v1/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [
              ...newMessages.map((m) => ({ role: m.role, content: m.content })),
            ],
          }),
        });
        
        if (!res.ok) {
          throw new Error(`Chat failed: ${res.status} ${res.statusText}`);
        }
        
        const data = await res.json();
        const assistantMessage: Message = { 
          role: "assistant" as const, 
          content: data.choices?.[0]?.message?.content || "[No response]" 
        };
        setMessages((msgs) => [...msgs, assistantMessage]);
        
        // Save assistant message for training
        await saveTrainingData(user.uid, {
          role: "assistant",
          content: assistantMessage.content
        });
        
        // Create or update chat session
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
          setChatSessions(prev => [newSession, ...prev]);
          setCurrentSessionId(newSession.id);
          
          // Save to Firebase
          await saveChatSession(user.uid, newSession);
        } else {
          // Update existing session
          const updatedSession = {
            ...chatSessions.find(s => s.id === currentSessionId)!,
            messages: updatedMessages, 
            title: generateSessionTitle(updatedMessages),
            updatedAt: new Date() 
          };
          
          setChatSessions(prev => prev.map(session => 
            session.id === currentSessionId ? updatedSession : session
          ));
          
          // Save to Firebase
          await saveChatSession(user.uid, updatedSession);
        }
      } catch (e) {
        console.error('Chat error:', e);
        setError(`Chat failed: ${e instanceof Error ? e.message : 'Unknown error'}. Please check if your backend is running at ${API_BASE}`);
      }
    }
    setLoading(false);
  }

  function handleImage(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setImagePreviewUrl(URL.createObjectURL(e.target.files[0]));
      setShowPlusMenu(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function renderMessage(msg: Message, i: number) {
    // Only show image preview for the current image before upload/analysis
    if (i === messages.length && imagePreviewUrl) {
      return (
        <div key={i} className="flex justify-end mb-4">
          <div className="max-w-xs">
            <Image src={imagePreviewUrl} alt="preview" width={320} height={240} className="rounded-lg shadow-sm" />
          </div>
        </div>
      );
    }
    if (msg.imageUrl) {
      console.log("Rendering image with URL:", msg.imageUrl);
      return (
        <div key={i} className="flex justify-end mb-4">
          <div className="max-w-xs">
            <Image src={msg.imageUrl} alt="uploaded" width={320} height={240} className="rounded-lg shadow-sm" />
          </div>
        </div>
      );
    }
    if (msg.imageResult) {
      const result = msg.imageResult;
      return (
        <div key={i} className="flex justify-start mb-4">
          <div className="max-w-3xl bg-gray-800 rounded-2xl shadow-sm border border-gray-700 p-4">
            <h3 className="text-lg font-medium text-gray-100 mb-3">Image Analysis Results</h3>
            
            {/* Classification */}
            {result.classification && result.classification.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Classification:</h4>
                <div className="flex flex-wrap gap-2">
                  {result.classification.map((item: { class: string; confidence: number }, idx: number) => (
                    <span key={idx} className="bg-blue-900 text-blue-200 px-2 py-1 rounded-full text-xs">
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
                    <span key={idx} className="bg-green-900 text-green-200 px-2 py-1 rounded-full text-xs">
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
                <div className={`p-3 rounded-lg ${result.caption.includes('[Note:') ? 'bg-yellow-900/30 border border-yellow-600' : 'bg-gray-700'}`}>
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
                <div className="bg-gray-700 p-3 rounded-lg">
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
    return (
      <div key={i} className={`flex mb-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
        <div
          className={`max-w-3xl rounded-2xl shadow-sm px-4 py-3 ${
            msg.role === "user"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-100 border border-gray-700"
          }`}
        >
          <div className="prose prose-sm max-w-none prose-invert">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
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
    <div className="min-h-screen bg-gray-900 flex">
      {/* Left Sidebar - Fixed */}
      {sidebarOpen && (
        <div className="w-64 bg-gray-900 border-r border-gray-700 flex flex-col fixed left-0 top-0 h-full z-10">
          <div className="p-4 border-b border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-white">
                <div className="w-8 h-8 bg-white rounded-sm flex items-center justify-center">
                  <svg className="w-6 h-6 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <span className="text-lg font-semibold">TraeGPT</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>
              <button 
                onClick={() => setSidebarOpen(false)}
                className="p-1 text-gray-400 hover:text-gray-300 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
          
          <div className="flex-1 p-4 space-y-1 overflow-y-auto">
            <button 
              onClick={handleNewChat}
              className="w-full p-3 text-gray-300 hover:text-white hover:bg-gray-800 rounded-lg flex items-center gap-3 transition-colors"
            >
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
                  className={`w-full p-3 text-gray-300 hover:text-white hover:bg-gray-800 rounded-lg flex items-center justify-between group cursor-pointer transition-colors ${
                    currentSessionId === session.id ? 'bg-gray-800 text-white' : ''
                  }`}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <span className="text-sm truncate">{session.title}</span>
                  </div>
                  <button
                    onClick={(e) => handleDeleteChatSession(session.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-400 transition-all"
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
        <div className="bg-gray-900 border-b border-gray-700 px-6 py-3 flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-4">
            {!sidebarOpen && (
              <button 
                onClick={() => setSidebarOpen(true)}
                className="p-2 text-gray-400 hover:text-gray-300 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}
            <h1 className="text-lg font-semibold text-white">TraeGPT</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setShowTools(!showTools)}
              className="flex items-center gap-1 text-gray-400 hover:text-gray-300 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span className="text-sm">Tools</span>
            </button>
          </div>
        </div>

        {/* Chat Container - Add top padding to account for fixed header */}
        <div className="flex-1 flex flex-col pt-0">
          <div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto px-6 py-6 space-y-4 pb-0"
          >
            {messages.length === 0 && (
              <div className="text-center text-gray-400 mt-20">
                <p className="text-lg">Ready when you are.</p>
              </div>
            )}
            {messages.map((msg, i) => renderMessage(msg, i))}
            {loading && (
              <div className="flex justify-start mb-4">
                <div className="bg-gray-800 rounded-2xl shadow-sm border border-gray-700 px-4 py-3">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-sm text-gray-400">AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area - Fixed to Bottom */}
          <div className="border-t border-gray-700 bg-gray-900 p-6 sticky bottom-0">
            <div className="max-w-4xl mx-auto">
              {/* Tools Panel */}
              {showTools && (
                <div className="mb-4 p-4 bg-gray-800 rounded-lg border border-gray-700">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">OCR Languages</label>
                      <input
                        type="text"
                        value={ocrLang}
                        onChange={e => setOcrLang(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white"
                        placeholder="en,es,fr"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">CLIP Categories</label>
                      <input
                        type="text"
                        value={categories}
                        onChange={e => setCategories(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white"
                        placeholder="cat,dog,car"
                        disabled={loading}
                      />
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="text-red-400 text-sm mb-3 text-center">{error}</div>
              )}

              <div className="flex items-end gap-3 bg-gray-800 rounded-2xl border border-gray-700 px-4 py-3">
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
                    <div className="absolute bottom-full left-0 mb-2 bg-gray-700 rounded-lg shadow-lg border border-gray-600 p-2 min-w-48">
                      <div className="space-y-1">
                        <button 
                          onClick={() => {
                            fileInputRef.current?.click();
                            setShowPlusMenu(false);
                          }}
                          className="w-full p-2 text-gray-300 hover:text-white hover:bg-gray-600 rounded flex items-center gap-2 text-sm"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                          Images
                        </button>
                        <button className="w-full p-2 text-gray-300 hover:text-white hover:bg-gray-600 rounded flex items-center gap-2 text-sm">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                          </svg>
                          Attachments
                        </button>
                        <button className="w-full p-2 text-gray-300 hover:text-white hover:bg-gray-600 rounded flex items-center gap-2 text-sm">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          Documents
                        </button>
                      </div>
                    </div>
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
                    onClick={sendMessage}
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
      </div>
      
      {/* Hidden file input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleImage}
        className="hidden"
      />
    </div>
  );
}
