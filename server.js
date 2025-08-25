const express = require('express');
const path = require('path');
const DeepGuardDB = require('./database');

const app = express();
const port = 3000;
const db = new DeepGuardDB();

app.use(express.static(__dirname));
app.use(express.json());

// API Routes
app.get('/api/scans', async (req, res) => {
    try {
        const scans = await db.getScans();
        res.json(scans);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/logs', async (req, res) => {
    try {
        const logs = await db.getLogs();
        res.json(logs);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/stats', async (req, res) => {
    try {
        const stats = await db.getStats();
        res.json(stats);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/scans', async (req, res) => {
    try {
        const { filename, confidence, threatLevel } = req.body;
        const scan = await db.addScan(filename, confidence, threatLevel);
        await db.addLog('success', `Scan completed: ${filename} (${confidence}% confidence)`);
        res.json(scan);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/logs', async (req, res) => {
    try {
        const { type, message } = req.body;
        const log = await db.addLog(type, message);
        res.json(log);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`DeepGuard server running at http://localhost:${port}`);
});

process.on('SIGINT', () => {
    db.close();
    process.exit(0);
});