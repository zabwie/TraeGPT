# Using Your PC as TraeGPT Server

## 🖥️ **Method 1: Local Network Only (Easiest)**

### For testing on your local network:
1. **Keep your current setup** (localhost:8000)2 **Deploy frontend to Vercel** with environment variable:
   ```
   NEXT_PUBLIC_API_BASE=http://localhost:8000
   ```
3. **Access from any device on your WiFi**

## 🌐 **Method 2ternet Access with Ngrok**

### Step1 Install Ngrok
```bash
# Download from https://ngrok.com/download
# Or install via npm
npm install -g ngrok
```

### Step 2: Start your backend
```bash
cd C:\Users\bugha\Desktop\TraeGPT
python traegpt_api.py
```

### Step3 Expose with Ngrok
```bash
ngrok http 8000### Step 4: Update Frontend
Use the ngrok URL in your Vercel environment:
```
NEXT_PUBLIC_API_BASE=https://abc123grok.io
```

## 🔧 **Method 3: Router Port Forwarding**

### Step1 Find your PC's IP
```bash
ipconfig
# Look for "IPv4 Address(usually 1920.168x.x)
```

### Step 2: Configure Router
1ccess router admin** (usually 192168.10.1*Find Port Forwarding** section3. **Add rule**:
   - External Port: 8000
   - Internal IP: Your PC's IP
   - Internal Port: 80   - Protocol: TCP

### Step3 Get Public IP
```bash
# Check your public IP
curl ifconfig.me
```

### Step 4: Update Frontend
```
NEXT_PUBLIC_API_BASE=http://YOUR_PUBLIC_IP:8000``

## 🚀 **Method 4: Dynamic DNS (Best for Production)**

### Step 1: Sign up for Dynamic DNS
- **No-IP** (free)
- **DuckDNS** (free)
- **DynDNS** (paid)

### Step2figure your router
- **Enable Dynamic DNS** in router settings
- **Enter your DDNS credentials**

### Step3 Port forward
- **External Port**:800- **Internal IP**: Your PCs IP
- **Internal Port**:8000### Step 4: Update Frontend
```
NEXT_PUBLIC_API_BASE=http://yourdomain.ddns.net:8000

## 🔒 **Security Considerations**

### **Firewall Setup**
```bash
# Windows Firewall - Allow port 8000
netsh advfirewall firewall add rule name="TraeGPT Backend" dir=in action=allow protocol=TCP localport=8000# **Basic Security**
1*Use HTTPS** (with ngrok or reverse proxy)2. **Limit access** to trusted IPs
3. **Regular updates** for your PC
4ng router password**

## 📊 **Performance Tips**

### **For Better Performance:**
1. **Use wired connection** instead of WiFi
2. **Close unnecessary programs** when serving
3Monitor CPU/memory** usage4r GPU acceleration** for AI models

### **Monitoring Commands:**
```bash
# Check if server is running
netstat -an | findstr :80 Monitor CPU usage
tasklist /fi "imagename eq python.exe"

# Check memory usage
wmic process where name="python.exe" get WorkingSetSize
```

## 🎯 **Recommended Setup**

### **For Development/Testing:**
- **Method 1** (Local Network) + **Ngrok** for external access

### **For Small Production:**
- **Method4** (Dynamic DNS) + **HTTPS reverse proxy**

### **For Maximum Control:**
- **Method3* (Port Forwarding) + **Custom domain**

## 💡 **Advantages of PC Server**

✅ **Completely free**  
✅ **Full control over models**  
✅ **No deployment complexity**  
✅ **Easy debugging**  
✅ **No API limits**  
✅ **Custom model training**  

## ⚠️ **Considerations**

❌ **PC must stay on 24/7**  
❌ **Depends on your internet**  
❌ **Security responsibility**  
❌ **No automatic scaling**  
❌ **Backup your own data**  

## 🚀 **Quick Start (Recommended)**
1 **Deploy frontend to Vercel**
2 ngrok for testing**
3. **Set up Dynamic DNS for production**
4. **Configure port forwarding**
5. **Update environment variables**

This gives you the best of both worlds: free hosting with full control! 