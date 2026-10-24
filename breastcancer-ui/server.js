/**
 * MammoXplain – Node.js API Gateway
 * Acts as a proxy between React frontend and Flask ML backend
 * 
 * Run: node server.js
 * Frontend: http://localhost:3000
 * This server: http://localhost:3000
 * Flask backend: http://localhost:5000
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5000';

// Multer for file uploads (stores in memory, passes to Flask)
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(cors({ origin: 'http://localhost:3000' }));
app.use(express.json());

// ── Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', gateway: 'MammoXplain Node Gateway', flaskUrl: FLASK_URL });
});

// ── Proxy all /api/* routes to Flask
app.use('/api', createProxyMiddleware({
  target: FLASK_URL,
  changeOrigin: true,
  on: {
    error: (err, req, res) => {
      console.error('Flask proxy error:', err.message);
      res.status(502).json({ error: 'Flask backend unavailable', detail: err.message });
    }
  }
}));

// ── Serve React build in production if available
const fs = require('fs');
const buildIndex = path.join(__dirname, 'build', 'index.html');
if (fs.existsSync(buildIndex)) {
  app.use(express.static(path.join(__dirname, 'build')));
  // Catch-all for client-side routing — ignore API routes
  app.get(/^\/(?!api).*/, (req, res) => {
    res.sendFile(buildIndex);
  });
} else {
  // Fallback to public/index.html when build is absent
  app.use(express.static(path.join(__dirname, 'public')));
  app.get(/^\/(?!api).*/, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`\n✅ MammoXplain Gateway running at http://localhost:${PORT}`);
  console.log(`   → Proxying API calls to Flask at ${FLASK_URL}`);
  console.log(`   → Serving React build from ./build\n`);
});
