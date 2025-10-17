import os
import mimetypes
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template
from typing import List, Dict, Tuple, Optional

class Report:

    DEFAULT_TEMPLATE = "templates/template.html"

    def __init__(self, default_template_path: str = DEFAULT_TEMPLATE):
        """
        Inicializa o Report com um template padrão.
        
        Args:
            default_template_path: Caminho para o template HTML.
        """
        self.default_template_path = default_template_path
        self._default_template = None
        self._validate_template_exists(default_template_path)

    def _validate_template_exists(self, template_path: str) -> None:
        """Valida se o template existe no caminho especificado."""
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"O Template {template_path} não foi encontrado.")

    def _load_template(self, template_path: Optional[str] = None) -> Template:
        """Carrega um template, seja o padrão ou um personalizado."""
        path = template_path if template_path else self.default_template_path
        self._validate_template_exists(path)
        
        template_dir = os.path.dirname(path) or '.'
        template_file_name = os.path.basename(path)
        
        env = Environment(loader=FileSystemLoader(template_dir))
        return env.get_template(template_file_name)

    def _scan_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Escaneia o diretório e classifica os arquivos por tipo.
        """
        classified_files = {
            'geo': [],
            'sensor': [],
            'images': []
        }
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        geo_extensions = ['.json', '.geojson', '.csv'] # Exemplo
        sensor_extensions = ['.csv', '.txt'] # Exemplo

        for item in os.listdir(directory_path):
            full_path = os.path.join(directory_path, item)
            if os.path.isfile(full_path):
                ext = os.path.splitext(item)[1].lower()
                
                if ext in image_extensions:
                    classified_files['images'].append(full_path)
                elif ext in geo_extensions:
                    # Adicionamos uma lógica simples para diferenciar geo de sensor
                    # (em um caso real, poderíamos ler o cabeçalho do arquivo)
                    if 'geo' in item.lower() or 'location' in item.lower():
                         classified_files['geo'].append(full_path)
                    else:
                        classified_files['sensor'].append(full_path)

        return classified_files

    def _process_files(self, classified_files: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
        """
        Processa os arquivos classificados e gera o conteúdo para cada seção.
        """
        sections_content = {
            'geo': None,
            'sensor': None,
            'images': None
        }
        
        summary = {}

        # Processa imagens
        if classified_files['images']:
            sections_content['images'] = classified_files['images']
            summary['Imagens'] = len(classified_files['images'])
            
        # Processa dados de sensores (placeholder)
        if classified_files['sensor']:
            # Aqui entraria a lógica para ler CSV/TXT, gerar gráficos com Matplotlib
            # e salvar como imagem, ou criar tabelas HTML com Pandas.
            # Por enquanto, apenas listamos os arquivos.
            html_content = "<ul>"
            for f in classified_files['sensor']:
                html_content += f"<li>{os.path.basename(f)}</li>"
            html_content += "</ul>"
            sections_content['sensor'] = html_content
            summary['Dados de Sensores'] = len(classified_files['sensor'])

        # Processa dados geográficos (placeholder)
        if classified_files['geo']:
            # Similarmente, aqui poderíamos usar GeoPandas/Folium para gerar um mapa
            # e salvá-lo como HTML.
            html_content = "<ul>"
            for f in classified_files['geo']:
                html_content += f"<li>{os.path.basename(f)}</li>"
            html_content += "</ul>"
            sections_content['geo'] = html_content
            summary['Dados Geográficos'] = len(classified_files['geo'])

        return summary, sections_content


    def generate_report_from_directory(self, directory_path: str, output_path: str, template_path: Optional[str] = None):
        """
        Gera um relatório HTML a partir de um diretório de dados.
        
        Args:
            directory_path: Caminho para o diretório com os dados.
            output_path: Caminho onde salvar o relatório HTML.
            template_path: Caminho para um template específico (opcional).
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"O caminho especificado não é um diretório: {directory_path}")

        print(f"Analisando o diretório: {directory_path}")
        classified_files = self._scan_directory(directory_path)

        print("Processando arquivos encontrados...")
        summary, sections = self._process_files(classified_files)

        # Carrega o template
        print("Carregando template...")
        template = self._load_template(template_path)

        # Define o título do relatório
        title = f"Relatório do Diretório '{os.path.basename(directory_path)}'"

        context = {
            "title": title,
            "summary": summary,
            "sections": sections,
            "data_geracao": datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }

        html_output = template.render(context)

        # Garante que o diretório de output exista
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        print(f"Relatório salvo com sucesso em: {output_path}")