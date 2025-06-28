// Global type declarations for VS Code extension

// Simplified types to avoid conflicts with built-in TypeScript definitions
declare namespace CodeGuard {
    interface RequestInit {
        body?: string | null;
        headers?: Record<string, string>;
        method?: string;
        signal?: any;
    }
    
    interface Response {
        ok: boolean;
        status: number;
        statusText: string;
        text(): Promise<string>;
        json(): Promise<any>;
    }
}