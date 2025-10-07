import os
import json
import base64
from markdown import markdown 
from datetime import datetime
from bs4 import BeautifulSoup # web scrapping
from jinja2 import Environment, FileSystemLoader, Template
from typing import List, Dict, Tuple, Optional

class VisualReport:

    DEFAULT_TEMPLATE = "templates/template1.html"

    def __init__(self, default_template_path: str = DEFAULT_TEMPLATE):
        """
        Inicializa o VisualReport com template padrão.
        
        Args:
            default_template_path: Caminho para template HTML. Usa o template padrão da classe se não especificado.
        """
        self.default_template_path = default_template_path
        self._default_template = None
        self._validate_template_exists(default_template_path)

    def _validate_template_exists(self, template_path: str) -> None:
        """
        Valida se o template existe no caminho especificado.
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(
                f"O Template {template_path} não foi encontrado.\n"
                f"Certifique-se de que o arquivo está incluído no pacote."
            )

    def _load_default_template(self) -> Template:
        """
        Carrega o template padrão (lazy loading).
        """
        if self._default_template is None:
            template_dir = os.path.dirname(self.default_template_path) or '.'
            template_file_name = os.path.basename(self.default_template_path)

            self.env = Environment(loader=FileSystemLoader(template_dir))
            self._default_template = self.env.get_template(template_file_name)
        
        return self._default_template

    def _load_custom_template(self, template_path: str) -> Template:
        """
        Carrega um template personalizado do usuário.
        """
        self._validate_template_exists(template_path)

        template_dir = os.path.dirname(template_path) or '.'
        template_file_name = os.path.basename(template_path)

        env = Environment(loader=FileSystemLoader(template_dir))

        return env.get_template(template_file_name)

    def _extract_title_from_markdown(self, html_content: str) -> Optional[str]:
        """
        Extrai título da primeira tag `<h1>` encontrada.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        h1_tag = soup.find('h1')

        return h1_tag.get_text().strip() if h1_tag else None

    def _process_markdown_cell(self, cell: Dict, first_markdown_h1: bool) -> Tuple[Dict, bool, Optional[str]]:
        """
        Processa uma célula markdown e retorna um componente e um status do título.
        """
        html_content = markdown("".join(cell['source']))
        title = None
        
        if first_markdown_h1:
            title = self._extract_title_from_markdown(html_content)
            if title:
                first_markdown_h1 = False
        
        component = {'type': 'markdown', 'data': html_content}

        return component, first_markdown_h1, title
    
    def _process_code_output(self, output: Dict) -> Optional[Dict]:
        """
        Processa o output de uma célula com código seguindo ordem de prioridade.
        """
        output_data = output.get('data', {})
        
        # Prioridade: HTML > PNG > Texto

        if 'text/html' in output_data:
            return {'type': 'html', 'data': "".join(output_data['text/html'])}
        
        elif 'image/png' in output_data:
            return {'type': 'image', 'data': output_data['image/png']}
        
        elif 'text/plain' in output_data:
            return {'type': 'text', 'data': "".join(output_data['text/plain'])}
        
        elif 'text' in output:  # Para output do tipo 'stream'
            return {'type': 'text', 'data': "".join(output['text'])}
        
        return None
    
    def parse_jupyter_notebook(self, jupyter_notebook_path: str) -> Tuple[str, List[Dict]]:
        """
        Lê e transforma um Jupyter notebook, extraindo seu título e seus componentes.
        """
        
        with open(jupyter_notebook_path, 'r', encoding='utf-8') as f:
            jn = json.load(f)

        # Título padrão será o nome do arquivo sem extensão
        title = os.path.splitext(os.path.basename(jupyter_notebook_path))[0]
        components = []
        first_markdown_h1 = True

        for cell in jn['cells']:
        # Para cada célula do notebook Jupyter (formato JSON), verificamos se é markdown e processamos o conteúdo.
            if cell['cell_type'] == 'markdown' and cell['source']:
                component, first_markdown_h1, new_title = self._process_markdown_cell(cell, first_markdown_h1)

                if new_title:
                    title = new_title

                components.append(component)

        # Se não é markdown, então iteramos sobre os outputs da célula
            elif cell['cell_type'] == 'code' and cell.get('outputs'):
                # Processa cada output, como imagens, textos ou htmls
                for output in cell['outputs']:
                    component = self._process_code_output(output)
                    if component:
                        components.append(component)
                        if component['type'] in ['html', 'image']:
                            break

        return title, components
    
    def generate_report(self, jupyter_notebook_path: str, output_path: str, template_path: Optional[str] = None) -> None:
        """
        Gera relatório HTML a partir de um Jupyter notebook.
        
        Args:
            jupyter_notebook_path: Caminho para o notebook .ipynb
            output_path: Caminho onde salvar o relatório HTML
            template_path: Caminho para um template específico (opcional)
        """
        
        # Extraímos o conteúdo do JN
        print(f"Lendo o notebook: {jupyter_notebook_path}")
        title, components = self.parse_jupyter_notebook(jupyter_notebook_path)

        # Carregamos o template
        print("Carregando template...")
        if template_path:
            template = self._load_custom_template(template_path)
            print(f"Usando template personalizado: {template_path}")
        else:
            template = self._load_default_template()
            print(f"Usando template padrão: {self.default_template_path}")

        # Variáveis que serão disponibilizadas para o template HTML.
        context = {
            "title": title,
            "components": components,
            "data_geracao": datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }

        html_output = template.render(context)

        # Garantimos que o diretório de output existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        print(f"Relatório salvo com sucesso em: {output_path}")
