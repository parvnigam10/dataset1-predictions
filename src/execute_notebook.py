"""
execute_notebook.py
Executes Dataset1_Predictor.ipynb cleanly and writes outputs back to the notebook file.
"""

import nbformat
from nbclient import NotebookClient

def execute_nb():
    notebook_path = 'notebooks/Dataset1_Predictor.ipynb'
    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': 'notebooks/'}})
    print("Executing notebook cells...")
    client.execute()
    
    print(f"Writing executed notebook back to {notebook_path}...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook execution completed successfully!")

if __name__ == '__main__':
    execute_nb()
