// Global type declarations for VS Code extension

// Define basic DOM types for axios compatibility
declare interface ArrayBufferView {
    buffer: ArrayBuffer;
    byteLength: number;
    byteOffset: number;
}

declare interface Blob {
    readonly size: number;
    readonly type: string;
}

declare interface FormData {
    append(name: string, value: string | Blob): void;
}

declare interface URLSearchParams {
    append(name: string, value: string): void;
}

type BodyInit = string | ArrayBuffer | ArrayBufferView | Blob | FormData | URLSearchParams;
type HeadersInit = Headers | Record<string, string> | Array<[string, string]>;

// Fix for axios RequestInit type issue
declare interface RequestInit {
    body?: BodyInit | null;
    headers?: HeadersInit;
    method?: string;
    signal?: AbortSignal | null;
}

// Headers interface
declare interface Headers {
    append(name: string, value: string): void;
    delete(name: string): void;
    get(name: string): string | null;
    has(name: string): boolean;
    set(name: string, value: string): void;
}

// Additional types for Node.js compatibility
declare interface AbortSignal {
    readonly aborted: boolean;
    addEventListener(type: string, listener: () => void): void;
    removeEventListener(type: string, listener: () => void): void;
}

// Additional DOM types if needed
declare interface Response {
    ok: boolean;
    status: number;
    statusText: string;
    text(): Promise<string>;
    json(): Promise<any>;
}