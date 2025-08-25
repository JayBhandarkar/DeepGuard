// Client-side database interface for DeepGuard
class DBClient {
    constructor() {
        this.baseURL = 'http://localhost:3000/api';
    }

    async getScans() {
        try {
            const response = await fetch(`${this.baseURL}/scans`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching scans:', error);
            return [];
        }
    }

    async getLogs() {
        try {
            const response = await fetch(`${this.baseURL}/logs`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching logs:', error);
            return [];
        }
    }

    async getStats() {
        try {
            const response = await fetch(`${this.baseURL}/stats`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching stats:', error);
            return { totalScans: 0, detectedCount: 0, accuracyRate: '0%', avgTime: '0s' };
        }
    }

    async addScan(filename, confidence, threatLevel) {
        try {
            const response = await fetch(`${this.baseURL}/scans`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, confidence, threatLevel })
            });
            return await response.json();
        } catch (error) {
            console.error('Error adding scan:', error);
            return null;
        }
    }

    async addLog(type, message) {
        try {
            const response = await fetch(`${this.baseURL}/logs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, message })
            });
            return await response.json();
        } catch (error) {
            console.error('Error adding log:', error);
            return null;
        }
    }
}

// Global instance
window.dbClient = new DBClient();