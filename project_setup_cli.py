#!/usr/bin/env python3
"""
CodeGuard ML Project Setup CLI
One-click environment setup for machine learning projects from command line.
"""

import argparse
import sys
import json
import requests
from pathlib import Path
from typing import Optional

class CodeGuardCLI:
    """Command-line interface for CodeGuard project setup."""
    
    def __init__(self, server_url: str = "https://codeguard.replit.app"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
    
    def list_templates(self) -> None:
        """List all available project templates."""
        try:
            response = self.session.get(f"{self.server_url}/templates")
            response.raise_for_status()
            data = response.json()
            
            print("🚀 CodeGuard ML Project Templates")
            print("=" * 40)
            
            for template in data['templates']:
                print(f"\n📦 {template['name']} ({template['title']})")
                print(f"   Framework: {template['framework']}")
                print(f"   Dependencies: {template['dependencies']}")
                print(f"   {template['description']}")
        
        except requests.RequestException as e:
            print(f"❌ Error fetching templates: {e}")
            sys.exit(1)
    
    def show_template(self, template_name: str) -> None:
        """Show detailed information about a specific template."""
        try:
            response = self.session.get(f"{self.server_url}/templates/{template_name}")
            response.raise_for_status()
            data = response.json()
            template = data['template']
            
            print(f"📋 Template Details: {template['name']}")
            print("=" * 50)
            print(f"Framework: {template['framework']}")
            print(f"Description: {template['description']}")
            
            print(f"\n📁 Files ({len(template['files'])}):")
            for file in template['files']:
                print(f"   • {file}")
            
            print(f"\n📂 Directories ({len(template['directories'])}):")
            for directory in template['directories']:
                print(f"   • {directory}/")
            
            print(f"\n📦 Dependencies ({len(template['dependencies'])}):")
            for dep in template['dependencies'][:10]:  # Show first 10
                print(f"   • {dep}")
            if len(template['dependencies']) > 10:
                print(f"   ... and {len(template['dependencies']) - 10} more")
            
            print(f"\n🛠️  Setup Commands:")
            for i, cmd in enumerate(template['setup_commands'], 1):
                print(f"   {i}. {cmd}")
        
        except requests.RequestException as e:
            print(f"❌ Error fetching template details: {e}")
            sys.exit(1)
    
    def preview_project(self, template_name: str) -> None:
        """Preview what will be created for a template."""
        try:
            response = self.session.post(
                f"{self.server_url}/templates/preview",
                json={"template": template_name}
            )
            response.raise_for_status()
            data = response.json()
            preview = data['preview']
            
            print(f"👀 Project Preview: {preview['template_name']}")
            print("=" * 40)
            print(f"Framework: {preview['framework']}")
            print(f"Files to create: {len(preview['files_to_create'])}")
            print(f"Directories to create: {len(preview['directories_to_create'])}")
            print(f"Dependencies: {preview['dependencies_count']}")
            
            print("\n📄 Files:")
            for file in preview['files_to_create']:
                print(f"   • {file}")
            
            print("\n📁 Directories:")
            for directory in preview['directories_to_create']:
                print(f"   • {directory}/")
            
            print("\n📦 Requirements Preview:")
            for req in preview['requirements_preview']:
                if req.strip():
                    print(f"   • {req}")
        
        except requests.RequestException as e:
            print(f"❌ Error previewing project: {e}")
            sys.exit(1)
    
    def create_project(self, template_name: str, project_path: str, 
                      config_file: Optional[str] = None, force: bool = False) -> None:
        """Create a new ML project from template."""
        
        # Check if directory exists
        path = Path(project_path)
        if path.exists() and not force:
            if path.is_dir() and list(path.iterdir()):
                print(f"❌ Directory {project_path} already exists and is not empty")
                print("   Use --force to overwrite or choose a different path")
                sys.exit(1)
        
        # Load custom configuration if provided
        custom_config = {}
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                print(f"📋 Loaded configuration from {config_file}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
        
        try:
            # Generate project
            print(f"🏗️  Creating {template_name} project at {project_path}...")
            
            response = self.session.post(
                f"{self.server_url}/templates/generate",
                json={
                    "template": template_name,
                    "project_path": project_path,
                    "config": custom_config
                }
            )
            response.raise_for_status()
            data = response.json()
            project = data['project']
            
            print("✅ Project created successfully!")
            print("=" * 30)
            print(f"Template: {project['template']}")
            print(f"Framework: {project['framework']}")
            print(f"Location: {project['project_path']}")
            print(f"Files created: {len(project['files_created'])}")
            print(f"Dependencies: {project['dependencies']}")
            
            print("\n🚀 Next Steps:")
            for i, step in enumerate(project['next_steps'], 1):
                print(f"   {i}. {step}")
            
            print(f"\n💡 Quick Start:")
            print(f"   cd {project_path}")
            print(f"   python -m venv venv")
            print(f"   source venv/bin/activate")
            print(f"   pip install -r requirements.txt")
            print(f"   python main.py")
        
        except requests.RequestException as e:
            print(f"❌ Error creating project: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Server response: {e.response.text}")
            sys.exit(1)
    
    def interactive_setup(self) -> None:
        """Interactive project setup wizard."""
        print("🧙 CodeGuard Interactive Project Setup")
        print("=" * 40)
        
        # Get available templates
        try:
            response = self.session.get(f"{self.server_url}/templates")
            response.raise_for_status()
            data = response.json()
            templates = data['templates']
        except requests.RequestException as e:
            print(f"❌ Error fetching templates: {e}")
            sys.exit(1)
        
        # Show templates
        print("\nAvailable templates:")
        for i, template in enumerate(templates, 1):
            print(f"   {i}. {template['title']} ({template['framework']})")
            print(f"      {template['description']}")
        
        # Get user choice
        while True:
            try:
                choice = int(input(f"\nSelect template (1-{len(templates)}): ")) - 1
                if 0 <= choice < len(templates):
                    selected_template = templates[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get project path
        default_path = f"./{selected_template['name'].lower().replace(' ', '_')}_project"
        project_path = input(f"Project path ({default_path}): ").strip()
        if not project_path:
            project_path = default_path
        
        # Confirm creation
        print(f"\n📋 Summary:")
        print(f"   Template: {selected_template['title']}")
        print(f"   Framework: {selected_template['framework']}")
        print(f"   Path: {project_path}")
        
        confirm = input("\nCreate project? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            self.create_project(selected_template['name'], project_path)
        else:
            print("❌ Project creation cancelled.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CodeGuard ML Project Setup - One-click environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  codeguard-setup list                          # List all templates
  codeguard-setup show pytorch_basic           # Show template details
  codeguard-setup preview rl_gym               # Preview project structure
  codeguard-setup create pytorch_basic ./my_project  # Create project
  codeguard-setup interactive                  # Interactive setup wizard
        """
    )
    
    parser.add_argument(
        '--server', 
        default="https://codeguard.replit.app",
        help="CodeGuard server URL"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all available templates')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show template details')
    show_parser.add_argument('template', help='Template name')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview project structure')
    preview_parser.add_argument('template', help='Template name')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new project')
    create_parser.add_argument('template', help='Template name')
    create_parser.add_argument('path', help='Project directory path')
    create_parser.add_argument('--config', help='Custom configuration JSON file')
    create_parser.add_argument('--force', action='store_true', help='Overwrite existing directory')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Interactive setup wizard')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = CodeGuardCLI(args.server)
    
    if args.command == 'list':
        cli.list_templates()
    elif args.command == 'show':
        cli.show_template(args.template)
    elif args.command == 'preview':
        cli.preview_project(args.template)
    elif args.command == 'create':
        cli.create_project(args.template, args.path, args.config, args.force)
    elif args.command == 'interactive':
        cli.interactive_setup()

if __name__ == "__main__":
    main()