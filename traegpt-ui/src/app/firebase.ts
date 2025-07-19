import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, User } from 'firebase/auth';
import { getFirestore, collection, doc, setDoc, getDocs, query, orderBy, deleteDoc, getDoc } from 'firebase/firestore';
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  imageUrl?: string;
  imageResult?: {
    caption?: string;
    object_detection?: Array<{
      class: string;
      confidence: number;
    }>;
    text_extraction?: Array<{
      text: string;
      confidence: number;
    }>;
    classification?: Array<{
      class: string;
      confidence: number;
    }>;
    analysis_time?: number;
  };
  fileUrl?: string;
  fileName?: string;
  fileType?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// Define a type for user preferences
export interface UserPreferences {
  userName?: string;
  userInterests?: string;
  answerStyle?: string;
  customPersonality?: string;
}

const firebaseConfig = {
  apiKey: "AIzaSyAIxQzSwBC1qD98vdt1hmJwAFYTYerYGsw",
  authDomain: "traegpt-3fc47.firebaseapp.com",
  projectId: "traegpt-3fc47",
  storageBucket: "traegpt-3fc47.firebasestorage.app",
  messagingSenderId: "886546096729",
  appId: "1:886546096729:web:6c71e84c5a45ca7b704748",
  measurementId: "G-4SSZ3TBEQH"
};

// Initialize Firebase
let app;
try {
  app = initializeApp(firebaseConfig);
} catch (error) {
  console.error('Firebase initialization error:', error);
  // Create a fallback config or handle the error
  throw new Error('Firebase configuration error. Please check your Firebase project settings.');
}

export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);

// Authentication functions
export const signInUser = async (): Promise<User> => {
  try {
    const result = await signInAnonymously(auth);
    return result.user;
  } catch (error) {
    console.error('Firebase sign-in error:', error);
    // If anonymous auth fails, we can try other methods or provide a fallback
    throw new Error(`Authentication failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

export const getCurrentUser = (): User | null => {
  return auth.currentUser;
};

// Chat session functions
export const saveChatSession = async (userId: string, session: ChatSession): Promise<void> => {
  const sessionRef = doc(db, 'users', userId, 'chatSessions', session.id);
  const safeMessages = session.messages.map((msg) => flattenNestedArrays(msg));
  await setDoc(sessionRef, {
    ...session,
    messages: safeMessages,
    createdAt: session.createdAt.toISOString(),
    updatedAt: session.updatedAt.toISOString()
  });
};

export const loadChatSessions = async (userId: string): Promise<ChatSession[]> => {
  const sessionsRef = collection(db, 'users', userId, 'chatSessions');
  const q = query(sessionsRef, orderBy('updatedAt', 'desc'));
  const querySnapshot = await getDocs(q);
  return querySnapshot.docs.map(docSnap => {
    const data = docSnap.data();
    return {
      ...data,
      createdAt: new Date(data.createdAt),
      updatedAt: new Date(data.updatedAt)
    } as ChatSession;
  });
};

export const deleteChatSession = async (userId: string, sessionId: string): Promise<void> => {
  const sessionRef = doc(db, 'users', userId, 'chatSessions', sessionId);
  await deleteDoc(sessionRef);
};

// Training data functions
export const saveTrainingData = async (userId: string, message: Message): Promise<void> => {
  const trainingRef = doc(db, 'trainingData', `${userId}_${Date.now()}`);
  await setDoc(trainingRef, {
    userId,
    message,
    timestamp: new Date().toISOString(),
    processed: false
  });
};

export async function uploadImageAndGetUrl(file: File, userId: string) {
  const storageRef = ref(storage, `chat_images/${userId}/${Date.now()}_${file.name}`);
  await uploadBytes(storageRef, file);
  return await getDownloadURL(storageRef);
}

// User preferences functions
// Save user preferences (personalization settings)
export const saveUserPreferences = async (userId: string, prefs: UserPreferences): Promise<void> => {
  const prefsRef = doc(db, 'users', userId, 'preferences', 'main');
  await setDoc(prefsRef, prefs, { merge: true });
};

// Load user preferences
export const loadUserPreferences = async (userId: string): Promise<UserPreferences> => {
  const prefsRef = doc(db, 'users', userId, 'preferences', 'main');
  const docSnap = await getDoc(prefsRef);
  return docSnap.exists() ? (docSnap.data() as UserPreferences) : {};
};

function deepFlattenArray(arr: unknown[]): unknown[] {
  return arr.reduce<unknown[]>((acc, val) =>
    Array.isArray(val)
      ? acc.concat(deepFlattenArray(val as unknown[]))
      : acc.concat([val])
  , []);
}

function flattenNestedArrays(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    // Recursively flatten all nested arrays
    return deepFlattenArray((obj as unknown[]).map(flattenNestedArrays));
  } else if (typeof obj === 'object' && obj !== null) {
    const newObj: Record<string, unknown> = {};
    for (const key in obj as Record<string, unknown>) {
      newObj[key] = flattenNestedArrays((obj as Record<string, unknown>)[key]);
    }
    return newObj;
  }
  return obj;
} 