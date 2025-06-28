// Global type declarations for VS Code extension
// DOM types are now available through tsconfig.json lib: ["DOM"]

declare namespace CodeGuard {
    // Extension-specific interfaces
    interface AuditOptions {
        level?: string;
        framework?: string;
        target?: string;
    }
    
    interface ProjectTemplate {
        name: string;
        description: string;
        framework: string;
        dependencies: number;
    }
}