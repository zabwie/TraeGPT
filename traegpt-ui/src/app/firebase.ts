import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, onAuthStateChanged, User } from 'firebase/auth';
import { getFirestore, collection, doc, setDoc, getDoc, getDocs, query, where, orderBy, deleteDoc } from 'firebase/firestore';

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
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);

// Authentication functions
export const signInUser = async () => {
  try {
    const result = await signInAnonymously(auth);
    return result.user;
  } catch (error) {
    console.error('Error signing in:', error);
    throw error;
  }
};

export const getCurrentUser = (): User | null => {
  return auth.currentUser;
};

// Chat session functions
export const saveChatSession = async (userId: string, session: any) => {
  try {
    const sessionRef = doc(db, 'users', userId, 'chatSessions', session.id);
    await setDoc(sessionRef, {
      ...session,
      createdAt: session.createdAt.toISOString(),
      updatedAt: session.updatedAt.toISOString()
    });
  } catch (error) {
    console.error('Error saving chat session:', error);
    throw error;
  }
};

export const loadChatSessions = async (userId: string) => {
  try {
    const sessionsRef = collection(db, 'users', userId, 'chatSessions');
    const q = query(sessionsRef, orderBy('updatedAt', 'desc'));
    const querySnapshot = await getDocs(q);
    
    const sessions = querySnapshot.docs.map(doc => {
      const data = doc.data();
      return {
        ...data,
        createdAt: new Date(data.createdAt),
        updatedAt: new Date(data.updatedAt)
      };
    });
    
    return sessions;
  } catch (error) {
    console.error('Error loading chat sessions:', error);
    return [];
  }
};

export const deleteChatSession = async (userId: string, sessionId: string) => {
  try {
    const sessionRef = doc(db, 'users', userId, 'chatSessions', sessionId);
    await deleteDoc(sessionRef);
  } catch (error) {
    console.error('Error deleting chat session:', error);
    throw error;
  }
};

// Training data functions
export const saveTrainingData = async (userId: string, message: any) => {
  try {
    const trainingRef = doc(db, 'trainingData', `${userId}_${Date.now()}`);
    await setDoc(trainingRef, {
      userId,
      message,
      timestamp: new Date().toISOString(),
      processed: false
    });
  } catch (error) {
    console.error('Error saving training data:', error);
    // Don't throw error for training data to avoid breaking user experience
  }
}; 