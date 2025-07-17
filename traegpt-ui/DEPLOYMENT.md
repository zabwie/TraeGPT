# TraeGPT Deployment Guide

## Prerequisites

1. **Firebase Project**: Make sure your Firebase project is set up with Firestore enabled
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **Backend API**: Deploy your FastAPI backend (traegpt_api.py) to a hosting service

## Deployment Steps

### 1. Deploy Backend API

First, deploy your FastAPI backend to a service like:
- Railway
- Render
- Heroku
- DigitalOcean App Platform

### 2. Deploy Frontend to Vercel

1. **Connect to Vercel**:
   ```bash
   # Install Vercel CLI
   npm i -g vercel
   
   # Login to Vercel
   vercel login
   ```

2. **Deploy the app**:
   ```bash
   cd traegpt-ui
   vercel
   ```

3. **Set Environment Variables**:
   In your Vercel dashboard, go to your project settings and add:
   ```
   NEXT_PUBLIC_API_BASE=https://your-backend-url.com
   ```

### 3. Firebase Security Rules

Update your Firestore security rules to allow authenticated users:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow users to read/write their own chat sessions
    match /users/{userId}/chatSessions/{sessionId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Allow training data collection
    match /trainingData/{docId} {
      allow write: if request.auth != null;
      allow read: if false; // Only allow writes, no reads
    }
  }
}
```

### 4. Custom Domain (Optional)

1. In Vercel dashboard, go to your project settings
2. Add your custom domain
3. Update DNS records as instructed

## Features

✅ **User Authentication**: Anonymous Firebase auth for each user
✅ **Cloud Storage**: All chat sessions stored in Firestore
✅ **Training Data**: User messages automatically saved for bot training
✅ **Persistent History**: Chat history persists across devices
✅ **Real-time Sync**: Changes sync across all user devices

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE` | Backend API URL | `https://traegpt-api.railway.app` |

## Training Data Collection

The app automatically saves all user messages and AI responses to the `trainingData` collection in Firestore. This data can be used to:

1. **Improve AI responses** by analyzing conversation patterns
2. **Identify common questions** and create better responses
3. **Train custom models** with real user interactions
4. **Analytics** to understand user behavior

## Security

- All data is user-scoped (users can only access their own data)
- Anonymous authentication prevents unauthorized access
- Training data is collected without personal information
- No sensitive data is stored in the app

## Monitoring

Check your Firebase console for:
- User authentication statistics
- Firestore usage and performance
- Training data collection metrics 