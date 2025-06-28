// src/globals.d.ts
export {};

declare global {
  interface RequestInit {
    /** body to be sent – string, ArrayBuffer, etc. */
    body?: string | ArrayBuffer | Blob | null;
    /** HTTP headers */
    headers?: Record<string, string>;
    /** GET, POST, PUT… */
    method?: string;
    /** cancellation */
    signal?: AbortSignal;
  }

  interface Response {
    ok: boolean;
    status: number;
    statusText: string;
    text(): Promise<string>;
    json(): Promise<any>;
  }
}
