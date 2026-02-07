# Render Deployment Guide

## Prerequisites
- GitHub account
- Render account (free tier)

## Deployment Steps

### 1. Prepare Your Repository

If you haven't already, initialize git and push to GitHub:

```powershell
cd travel-mcp-server
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/travel-mcp-server.git
git push -u origin main
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New →** **Web Service**
3. Select **Build and deploy from a Git repository**
4. Click **Connect GitHub** and authorize Render
5. Select your `travel-mcp-server` repository
6. Fill in these details:
   - **Name**: `travel-mcp`
   - **Environment**: `Node`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npm start`
   - **Plan**: Select the free tier
7. Under **Environment**, add variables if needed:
   - `PORT`: `3000`
   - `NODE_ENV`: `production`
8. Click **Create Web Service**

### 3. Get Your URL

Once deployed, you'll see a URL like:
```
https://travel-mcp-XXXX.onrender.com
```

### 4. Use in MCP Client

Use this URL in your MCP client configuration:
```
https://travel-mcp-XXXX.onrender.com/sse
```

## Auto-Deploy

Every time you push to GitHub, Render automatically redeploys your server. No need to manually redeploy!

## Monitoring

- View logs in Render dashboard
- Check deployment status in real-time
- Restart service if needed from dashboard

## Troubleshooting

If deployment fails:
1. Check the **Logs** tab in Render dashboard
2. Common issues:
   - Missing environment variables
   - Port binding issues (already fixed in this config)
   - TypeScript compilation errors

## Free Tier Limits

- 750 hours/month (enough for always-on service)
- Auto-spins down after 15 min of inactivity (you can upgrade to prevent this)
- 0.5GB RAM (sufficient for MCP server)
