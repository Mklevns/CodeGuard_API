// Global type declarations for VS Code extension

// Fix for axios RequestInit type issue
declare interface RequestInit {
    body?: BodyInit | null;
    headers?: HeadersInit;
    method?: string;
    signal?: AbortSignal | null;
}

// Additional DOM types if needed
declare interface Response {
    ok: boolean;
    status: number;
    statusText: string;
    text(): Promise<string>;
    json(): Promise<any>;
}