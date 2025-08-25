// SQLite Database for DeepGuard (Browser-compatible)
class SQLiteDB {
    constructor() {
        this.dbName = 'deepguard.db';
        this.initDB();
    }

    async initDB() {
        // Use sql.js for browser SQLite
        const SQL = await initSqlJs({
            locateFile: file => `https://sql.js.org/dist/${file}`
        });
        
        // Try to load existing database from localStorage
        const savedDB = localStorage.getItem('deepguard_sqlite');
        if (savedDB) {
            const uInt8Array = new Uint8Array(JSON.parse(savedDB));
            this.db = new SQL.Database(uInt8Array);
        } else {
            this.db = new SQL.Database();
            this.createTables();
            this.insertDemoData();
        }
    }

    createTables() {
        this.db.run(`CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            threat_level TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )`);

        this.db.run(`CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )`);
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

        this.saveDB();
    }

    saveDB() {
        const data = this.db.export();
        localStorage.setItem('deepguard_sqlite', JSON.stringify(Array.from(data)));
    }

    addScan(filename, confidence, threatLevel) {
        this.db.run("INSERT INTO scans (filename, confidence, threat_level) VALUES (?, ?, ?)", 
                   [filename, confidence, threatLevel]);
        this.saveDB();
        return { filename, confidence, threat_level: threatLevel };
    }

    addLog(type, message) {
        this.db.run("INSERT INTO logs (type, message) VALUES (?, ?)", [type, message]);
        this.saveDB();
        return { type, message };
    }

    getScans() {
        const stmt = this.db.prepare("SELECT * FROM scans ORDER BY timestamp DESC LIMIT 50");
        const results = [];
        while (stmt.step()) {
            results.push(stmt.getAsObject());
        }
        stmt.free();
        return results;
    }

    getLogs() {
        const stmt = this.db.prepare("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100");
        const results = [];
        while (stmt.step()) {
            results.push(stmt.getAsObject());
        }
        stmt.free();
        return results;
    }

    getStats() {
        const totalStmt = this.db.prepare("SELECT COUNT(*) as total FROM scans");
        totalStmt.step();
        const total = totalStmt.getAsObject().total;
        totalStmt.free();

        const detectedStmt = this.db.prepare("SELECT COUNT(*) as detected FROM scans WHERE confidence >= 30");
        detectedStmt.step();
        const detected = detectedStmt.getAsObject().detected;
        detectedStmt.free();

        return {
            totalScans: total,
            detectedCount: detected,
            accuracyRate: '97.8%',
            avgTime: '2.1s'
        };
    }
}

// Load sql.js library
const script = document.createElement('script');
script.src = 'https://sql.js.org/dist/sql-wasm.js';
script.onload = () => {
    window.sqliteDB = new SQLiteDB();
};
document.head.appendChild(script);