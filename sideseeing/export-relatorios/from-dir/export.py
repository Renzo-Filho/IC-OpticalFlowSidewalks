import os
import json
import base64
from datetime import datetime
from bs4 import BeautifulSoup # web scrapping
from jinja2 import Environment, FileSystemLoader, Template
from typing import List, Dict, Tuple, Optional

class Report:

    DEFAULT_TEMPLATE = "templates/template1.html"

    def __init__(self, default_template_path: str = DEFAULT_TEMPLATE):
        """
        Inicializa o Report com template padrão.
        
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
    
    def _process_data(self):
        pass

    def generate_report(self, template_path: Optional[str] = None) -> None:

        # ...

        # Carregamos o template
        print("Carregando template...")
        if template_path:
            template = self._load_custom_template(template_path)
            print(f"Usando template personalizado: {template_path}")
        else:
            template = self._load_default_template()
            print(f"Usando template padrão: {self.default_template_path}")

        # ...