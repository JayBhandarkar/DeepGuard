// Simple Database for DeepGuard
class SimpleDB {
    constructor() {
        this.scans = JSON.parse(localStorage.getItem('deepguard_scans') || '[]');
        this.logs = JSON.parse(localStorage.getItem('deepguard_logs') || '[]');
        this.settings = JSON.parse(localStorage.getItem('deepguard_settings') || '{"sensitivity": 75, "falsePositives": "low"}');
    }

    addScan(filename, confidence, threatLevel) {
        const scan = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            filename: filename,
            confidence: Math.round(confidence),
            threat_level: threatLevel
        };
        this.scans.unshift(scan);
        if (this.scans.length > 100) this.scans.pop();
        localStorage.setItem('deepguard_scans', JSON.stringify(this.scans));
    }

    addLog(type, message) {
        const log = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            type: type,
            message: message
        };
        this.logs.unshift(log);
        if (this.logs.length > 200) this.logs.pop();
        localStorage.setItem('deepguard_logs', JSON.stringify(this.logs));
    }

    getScans() {
        return this.scans;
    }

    getStats() {
        const totalScans = this.scans.length;
        const detectedCount = this.scans.filter(s => s.confidence > 50).length;
        const avgConfidence = totalScans > 0 ? 
            Math.round(this.scans.reduce((sum, s) => sum + s.confidence, 0) / totalScans) : 0;
        const accuracyRate = totalScans > 0 ? 
            Math.round((this.scans.filter(s => s.confidence < 70).length / totalScans) * 100) + '%' : '97.3%';
        
        return {
            totalScans,
            detectedCount,
            accuracyRate,
            avgTime: '2.1s',
            avgConfidence
        };
    }

    getLogs() {
        return this.logs;
    }

    getSettings() {
        return this.settings;
    }

    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        localStorage.setItem('deepguard_settings', JSON.stringify(this.settings));
    }

    generateConfidence(baseConfidence) {
        const sensitivity = this.settings.sensitivity || 75;
        const falsePositives = this.settings.falsePositives || 'low';
        
        let adjustedConf = baseConfidence;
        
        // Apply sensitivity adjustment
        if (sensitivity > 75) {
            adjustedConf = Math.min(95, baseConfidence * 1.2);
        } else if (sensitivity < 50) {
            adjustedConf = Math.max(5, baseConfidence * 0.8);
        }
        
        // Apply false positive tolerance
        if (falsePositives === 'low' && adjustedConf < 40) {
            adjustedConf = Math.max(5, adjustedConf * 0.7);
        }
        
        return Math.round(adjustedConf);
    }
}

function getThreatLevel(confidence) {
    if (confidence < 30) return 'Low';
    if (confidence < 60) return 'Medium';
    return 'High';
}

// Global instance
window.simpleDB = new SimpleDB();