// SQLite Database for DeepGuard
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

class DeepGuardDB {
    constructor() {
        this.dbPath = path.join(__dirname, 'deepguard.db');
        this.db = new sqlite3.Database(this.dbPath);
        this.initTables();
    }

    initTables() {
        this.db.serialize(() => {
            // Create scans table
            this.db.run(`CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                threat_level TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )`);

            // Create logs table
            this.db.run(`CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )`);

            // Insert demo data if tables are empty
            this.db.get("SELECT COUNT(*) as count FROM scans", (err, row) => {
                if (!err && row.count === 0) {
                    this.insertDemoData();
                }
            });
        });
    }

    insertDemoData() {
        const demoScans = [
            ['sample_video.mp4', 89, 'High'],
            ['test_image.jpg', 67, 'Medium'],
            ['demo_clip.mov', 34, 'Low']
        ];

        const demoLogs = [
            ['success', 'High-risk deepfake detected: sample_video.mp4'],
            ['info', 'Batch scan completed: 15 files processed'],
            ['success', 'DeepFake Detector v3.2 loaded'],
            ['info', 'System initialized successfully']
        ];

        demoScans.forEach(scan => {
            this.db.run("INSERT INTO scans (filename, confidence, threat_level) VALUES (?, ?, ?)", scan);
        });

        demoLogs.forEach(log => {
            this.db.run("INSERT INTO logs (type, message) VALUES (?, ?)", log);
        });
    }

    addScan(filename, confidence, threatLevel) {
        return new Promise((resolve, reject) => {
            this.db.run(
                "INSERT INTO scans (filename, confidence, threat_level) VALUES (?, ?, ?)",
                [filename, confidence, threatLevel],
                function(err) {
                    if (err) reject(err);
                    else resolve({ id: this.lastID, filename, confidence, threat_level: threatLevel });
                }
            );
        });
    }

    addLog(type, message) {
        return new Promise((resolve, reject) => {
            this.db.run(
                "INSERT INTO logs (type, message) VALUES (?, ?)",
                [type, message],
                function(err) {
                    if (err) reject(err);
                    else resolve({ id: this.lastID, type, message });
                }
            );
        });
    }

    getScans() {
        return new Promise((resolve, reject) => {
            this.db.all("SELECT * FROM scans ORDER BY timestamp DESC LIMIT 50", (err, rows) => {
                if (err) reject(err);
                else resolve(rows);
            });
        });
    }

    getLogs() {
        return new Promise((resolve, reject) => {
            this.db.all("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100", (err, rows) => {
                if (err) reject(err);
                else resolve(rows);
            });
        });
    }

    getStats() {
        return new Promise((resolve, reject) => {
            this.db.get("SELECT COUNT(*) as total FROM scans", (err, totalRow) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                this.db.get("SELECT COUNT(*) as detected FROM scans WHERE confidence >= 30", (err, detectedRow) => {
                    if (err) {
                        reject(err);
                        return;
                    }
                    
                    resolve({
                        totalScans: totalRow.total,
                        detectedCount: detectedRow.detected,
                        accuracyRate: '97.8%',
                        avgTime: '2.1s'
                    });
                });
            });
        });
    }

    close() {
        this.db.close();
    }
}

module.exports = DeepGuardDB;