"""
ACL Anthology Classification Paper Cartography Script (Optimized)

This script collects and analyzes classification papers from
ACL, NAACL, EACL, CoNLL, EMNLP, COLING and LREC conferences
"""

from acl_anthology import Anthology
import pandas as pd
from typing import List, Dict, Iterable, Optional
from collections import defaultdict
import unicodedata
import re
import numpy as np


class ACLCartographer:
    """Class for mapping classification papers in ACL Anthology"""
    
    # Conferences of interest (venue prefixes in ACL Anthology)
    CONFERENCES = {
        'acl': ['acl'],
        'naacl': ['naacl'],
        'eacl': ['eacl'],
        'conll': ['conll'],
        'emnlp': ['emnlp'],
        'coling': ['coling'],
        'lrec': ['lrec'],
        "findings": ["findings"]
    }
    
    # Keywords for identifying classification types
    CLASSIFICATION_KEYWORDS = [
        'classification', 'classifier', 'categorization', 'labeling',
        'sentiment analysis', 'multi-class', 'multiclass', 'multi-label'
    ]
    
    # Classification types
    CLASSIFICATION_TYPES = {
        'binary': [
            'binary classification', 'binary', 'two-class', 'yes/no', 'positive vs negative',
            'sentiment polarity', 'spam', 'fake news', 'toxic', 'toxicity'
        ],
        'multi-class': [
            'multi-class', 'multiclass', 'multiple classes', 'topic classification',
            'intent classification', 'emotion classification', 'aspect classification',
            'category classification', 'stance detection', 'multiple categories', 'predicted class', 'class prediction', 'categories'
        ],
        'multi-label': [
            'multi-label', 'multilabel', 'multiple labels', 'hierarchical classification',
            'multi-topic classification', 'joint classification', 'labeling', 'labels'
        ]
    }

    
    # Classification domains
    DOMAINS = {
        'sentiment': ['sentiment', 'opinion', 'polarity', 'emotion', 'review', 'emotional'],
        'topic': ['topic', 'subject', 'category'],
        'intent': ['intent', 'intention'],
        'stance': ['stance detection', 'interpretation', 'opinion'],
        'toxicity': ['toxicity','toxic', 'hate speech', 'offensive'],
        'fake_news': ['fake news', 'misinformation', 'fact'],
        'ner': ['named entity', 'ner'],
        'other': []
    }

    MODELS = {
        'transformer': ['transformer', 'bert', 'roberta', 't5', 'gpt', 'gpt-2', 'gpt-3', 'gpt-neo', 'gpt-j', 'gpt-4', 'mpnet', 'clip', 'vilt', 'visualbert', 'minilm', 'spanbert', 'llama', 'llama-2', 'llama-3', 'llama3', 'llama-3.1', 'llama-3.2', 'xlnet', 'distilbert', 'albert', 'electra', 'deberta', 'deberta-v3', 'ernie', 'ernie-2.0', 'bart', 'turing-nlg', 'self-attention', 'mistral', 'mixtral', 'xlm', 'xlm-r', 'longformer', 'bigbird'],
        'rnn': ['rnn', 'gru', 'birnn', 'bigru', 'hrnn', 'rnn-crf', 'tree-rnn', 'gru-capsule','recurrent neural network'],
        'cnn': ['cnn', 'convolutional neural network', 'tcn', 'textcnn', 'dcnn', 'rcnn'],
        'lstm': ['lstm', 'ulmfit', 'elmo', 'flair embeddings', 'bi-lstm', 'bilstm', 'awd-lstm', 'han'],
        'classical': ['svm', 'svc', 'linear regression', 'logistic regression', 'naive bayes', 'random forest', 'xgboost', 'lightgbm', 'gaussian mixture model', 'gmm', 'k-means', 'bayesian', 'clustering', 'decision tree',],
        'graph_based': ['graph', 'gnn', 'gcn', 'gat', 'hgat', 'h-gnn', 'r-gcn', 'textgcn', 'graphsage', 'hypergraph'],
        'other_dl': ['autoencoder', 'deep learning', 'gan', 'vae', 'mixture of experts', 'neural network', 'adversarial network', 'routing network', 'energy-based model', 'mlp', 'feedforward network', 'deep neural']
    }
    
    def __init__(self, start_year: int = 2020, end_year: int = 2025):
        """
        Initialize the cartographer
        
        Args:
            start_year: Starting year
            end_year: Ending year
        """
        self.start_year = start_year
        self.end_year = end_year
        print("Initializing ACL Anthology from official repository...")
        self.anthology = Anthology.from_repo()
        print("✓ Anthology loaded successfully")
        print(f"  Total volumes: {len(list(self.anthology.volumes()))}")
        
        # Load benchmark data ONCE during initialization
        self._benchmark_data = None
        self._benchmark_loaded = False
        
        # Pre-compile regex patterns
        self._metric_patterns = {
            'accuracy': re.compile(r'accuracy[:\s]+(\d+\.?\d*)%?'),
            'f1': re.compile(r'f1[:\s-]+(\d+\.?\d*)%?'),
            'precision': re.compile(r'precision[:\s]+(\d+\.?\d*)%?'),
            'recall': re.compile(r'recall[:\s]+(\d+\.?\d*)%?'),
        }
        
    def _load_benchmark_data(self):
        """Load benchmark data once and cache it"""
        if self._benchmark_loaded:
            return
            
        print("Loading benchmark data (one-time operation)...")
        try:
            df = pd.read_parquet(
                "hf://datasets/pwc-archive/datasets/data/train-00000-of-00001.parquet"
            )
            
            print(f"  Total datasets loaded: {len(df)}")
            
            # Filter to text-classification datasets only
            mask = df.apply(self._contains_text_classification, axis=1)
            self._benchmark_data = df[mask].copy()
            
            print(f"  Text classification datasets found: {len(self._benchmark_data)}")
            
            if len(self._benchmark_data) > 0:
                # Pre-process benchmark names for faster matching
                self._benchmark_data['all_names'] = self._benchmark_data.apply(
                    lambda row: self._collect_name_aliases(row), axis=1
                )
                
                # Show some example benchmark names
                sample_names = self._benchmark_data['name'].head(10).tolist()
                print(f"  Example benchmarks: {', '.join(sample_names[:5])}")
            
            print(f"  ✓ Loaded {len(self._benchmark_data)} text classification benchmarks")
        except Exception as e:
            print(f"  Warning: Could not load benchmark data: {e}")
            import traceback
            traceback.print_exc()
            self._benchmark_data = pd.DataFrame()
        
        self._benchmark_loaded = True
        
    def _extract_paper_metadata(self, paper, year: int = None) -> Optional[Dict]:
        """
        Extract metadata from a paper object
        
        Args:
            paper: Paper object from acl-anthology
            year: Year (if already known from volume)
            
        Returns:
            Dictionary with paper metadata
        """
        try:
            # Use provided year if available, otherwise extract from paper
            if year is None:
                year = getattr(paper, 'year', None)
                if year is None and hasattr(paper, 'parent_volume') and paper.parent_volume:
                    year = getattr(paper.parent_volume, 'year', None)
                
                # Convert year to int if it's a string
                if year is not None and isinstance(year, str):
                    try:
                        year = int(year)
                    except (ValueError, TypeError):
                        year = None
            
            # Extract basic info
            metadata = {
                'paper_id': str(paper.id),
                'title': str(paper.title),
                'abstract': str(paper.abstract) if paper.abstract else "",
                'year': year,
                'authors': [str(author.full) if hasattr(author, 'full') else str(author) for author in paper.authors] if paper.authors else [],
                'venue': str(paper.parent_volume.title) if hasattr(paper, 'parent_volume') and paper.parent_volume else "",
                'citations': 0,  # Will be populated if available
            }
                
            return metadata
            
        except Exception as e:
            print(f"    Error extracting metadata: {e}")
            return None
    
    def is_classification_paper(self, paper: Dict) -> bool:
        """
        Determine if a paper is about classification
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            True if it's a classification paper
        """
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Check if at least one keyword is present
        return any(keyword.lower() in text for keyword in self.CLASSIFICATION_KEYWORDS)
    
    def classify_paper_type(self, paper: Dict) -> List[str]:
        """
        Identify classification type
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            List of classification types
        """
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        types = []
        
        for type_name, keywords in self.CLASSIFICATION_TYPES.items():
            if any(keyword.lower() in text for keyword in keywords):
                types.append(type_name)

        # --- Heuristic fallback ---
        if not types:
            if 'classification' in text and any(
                k in text for k in ['sentiment', 'topic', 'intent', 'emotion', 'stance', 'aspect']
            ):
                types.append('multi-class')
            elif 'classification' in text and any(k in text for k in ['binary', 'yes/no', 'positive']):
                types.append('binary')

        return types if types else ['unspecified']
    
    def classify_domain(self, paper: Dict) -> List[str]:
        """
        Identify classification domain
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            List of domains
        """
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        domains = []
        
        for domain_name, keywords in self.DOMAINS.items():
            if domain_name == 'other':
                continue
            if any(keyword.lower() in text for keyword in keywords):
                domains.append(domain_name)
                    
        return domains if domains else ['other']
    
    def classify_model_type(self, paper: Dict) -> List[str]:
        """Identify the machine learning model/architecture used."""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        models = []
        
        for model_name, keywords in self.MODELS.items():
            for keyword in keywords:
                # Use a word boundary \b to prevent partial matches like 'roberts' -> 'bert'
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    models.append(model_name)
                    break
        
        return list(set(models)) if models else ['unspecified']

    def _normalize(self, s: str) -> str:
        """Normalize string for matching"""
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-zA-Z0-9]+", " ", s).strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _as_iter(self, x) -> Iterable[str]:
        """Convert to iterable of strings"""
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        try:
            return [y for y in x if isinstance(y, str)]
        except TypeError:
            return []

    def _contains_text_classification(self, row) -> bool:
        """Check if row contains text classification task"""
        # The 'tasks' column contains arrays of dictionaries like:
        # [{'task': 'Text Classification', 'url': '...'}, ...]
        
        if 'tasks' not in row or row['tasks'] is None:
            return False
        
        tasks = row['tasks']
        
        # Handle numpy arrays
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        
        # Handle if it's not a list
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]
        
        # Check each task
        for task_item in tasks:
            task_name = None
            
            # Extract task name from dictionary or string
            if isinstance(task_item, dict):
                task_name = task_item.get('task', '')
            elif isinstance(task_item, str):
                task_name = task_item
            else:
                continue
            
            if not task_name or not isinstance(task_name, str):
                continue
            
            task_lower = task_name.lower()
            
            # Check for text classification related terms
            if any(term in task_lower for term in [
                'text classification', 
                'sentiment',
                'text categorization',
                'document classification',
                'topic classification',
                'intent classification',
                'multi label text classification',
                'multi-label text classification',
                'token classification',
                'sequence classification'
            ]):
                return True
        
        return False

    @staticmethod
    def _collect_name_aliases(row) -> List[str]:
        """Collect all name aliases for a benchmark"""
        name_candidates = []
        for c in ["name", "full_name"]:
            if c in row and isinstance(row[c], str) and row[c].strip():
                name_candidates.append(row[c].strip())

        # Deduplicate while preserving order
        seen = set()
        out = []
        for n in name_candidates:
            if n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out

    def extract_benchmarks(self, paper: Dict) -> List[str]:
        """
        Return only the text-classification benchmarks that are explicitly mentioned
        in the paper's title/abstract (case-insensitive, robust to punctuation).
        """
        # Ensure benchmark data is loaded
        if not self._benchmark_loaded:
            self._load_benchmark_data()
        
        if self._benchmark_data is None or len(self._benchmark_data) == 0:
            return []
        
        # Build searchable text from paper
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        text_norm = self._normalize(text)

        # Check each benchmark
        mentioned = []
        for _, row in self._benchmark_data.iterrows():
            names = row.get('all_names', [])
            if not names:
                continue

            primary = names[0]
            found = False
            
            for alias in names:
                alias_clean = alias.strip()
                if not alias_clean:
                    continue

                # Word boundary check
                pat = r"\b" + re.escape(alias_clean) + r"\b"
                if re.search(pat, text, flags=re.IGNORECASE):
                    found = True
                    break

                # Normalized check
                alias_norm = self._normalize(alias_clean)
                if alias_norm and alias_norm in text_norm:
                    found = True
                    break

            if found:
                mentioned.append(primary)

        # Deduplicate
        seen = set()
        out = []
        for name in mentioned:
            if name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)

        return out
    
    def extract_performance_metrics(self, paper: Dict) -> Dict:
        """
        Extract performance metrics mentioned in abstract
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Dictionary with performance info
        """
        abstract = paper.get('abstract', '').lower()
        metrics = {}
        
        for metric_name, pattern in self._metric_patterns.items():
            match = pattern.search(abstract)
            if match:
                metrics[metric_name] = float(match.group(1))
                
        return metrics
    
    def run_cartography(self, debug: bool = False) -> pd.DataFrame:
        """
        Run complete cartography
        
        Args:
            debug: Enable debug output
        
        Returns:
            DataFrame with all results
        """
        print("\nStarting ACL cartography...")
        print(f"Period: {self.start_year}-{self.end_year}")
        print(f"Conferences: {', '.join(self.CONFERENCES.keys())}\n")
        
        # Load benchmark data once before processing
        self._load_benchmark_data()
        
        all_papers = []
        volumes_checked = 0
        volumes_matched = 0
        papers_checked = 0
        
        # Pre-compile volume ID patterns
        # New format: 2020.acl-main
        new_format_pattern = re.compile(r'\d{4}\.([a-z]+)')
        # Old format: P07-1, A00-1, etc. - letter prefix indicates conference
        old_format_pattern = re.compile(r'^([A-Z])\d{2}-')
        
        # Map old format letters to conferences
        old_format_map = {
            'P': 'acl',      # ACL (Proceedings)
            'A': 'acl',      # ACL (Applied)  
            'C': 'coling',   # COLING
            'D': 'emnlp',    # EMNLP (Demonstrations)
            'E': 'eacl',     # EACL
            'N': 'naacl',    # NAACL
            'H': 'naacl',    # HLT-NAACL
            'W': None,       # Workshops (skip these)
            'O': None,       # Other events (skip)
            'J': None,       # Journal (skip)
            'L': 'lrec',     # LREC
            'I': None,       # IJCNLP (not in our list)
            'S': None,       # Seminars (skip)
            'T': None,       # Tutorials (skip)
            'U': None,       # Unknown (skip)
            'R': None,       # Resources (skip)
            'Y': None,       # Other year-based (skip)
        }
        
        # Flatten conference prefixes for faster lookup (new format)
        conf_prefix_map = {}
        for conf_key, conf_prefixes in self.CONFERENCES.items():
            for prefix in conf_prefixes:
                conf_prefix_map[prefix] = conf_key
        
        # Track year range found
        years_found = set()
        
        # Iterate through all volumes in the anthology
        for volume in self.anthology.volumes():
            volumes_checked += 1
            volume_id = str(volume.full_id)  # Get volume_id early for error handling
            
            # Check if this volume matches our criteria
            try:
                # Get year directly from volume object
                year = getattr(volume, 'year', None)
                
                # Convert year to int if it's a string
                if isinstance(year, str):
                    try:
                        year = int(year)
                    except (ValueError, TypeError):
                        continue
                
                # Check if year is in range
                if not (self.start_year <= year <= self.end_year):
                    continue
                
                years_found.add(year)
                
                # Extract conference from volume ID (support both old and new formats)
                conf_name = None
                
                # Try new format first: 2020.acl-main
                new_match = new_format_pattern.match(volume_id)
                if new_match:
                    conf_prefix = new_match.group(1)
                    conf_name = conf_prefix_map.get(conf_prefix)
                else:
                    # Try old format: P07-1, A00-1, etc.
                    old_match = old_format_pattern.match(volume_id)
                    if old_match:
                        letter_code = old_match.group(1)
                        conf_name = old_format_map.get(letter_code)
                
                if not conf_name:
                    continue
                
                volumes_matched += 1
                if debug and volumes_matched <= 3:
                    print(f"  Processing volume: {volume_id} (year: {year})")
                
                # Process papers in this volume
                volume_papers = 0
                for paper in volume.papers():
                    papers_checked += 1
                    paper_data = self._extract_paper_metadata(paper, year)  # Pass year from volume
                    
                    if paper_data and self.is_classification_paper(paper_data):
                        paper_data['conference'] = conf_name
                        paper_data['classification_types'] = self.classify_paper_type(paper_data)
                        paper_data['domains'] = self.classify_domain(paper_data)
                        paper_data['model_types'] = self.classify_model_type(paper_data)
                        paper_data['benchmarks'] = self.extract_benchmarks(paper_data)
                        paper_data['num_benchmarks'] = len(paper_data['benchmarks'])
                        paper_data['performance'] = self.extract_performance_metrics(paper_data)
                        
                        all_papers.append(paper_data)
                        volume_papers += 1
                
                if volume_papers > 0:
                    print(f"  {volume_id}: {volume_papers} classification papers")
                    
            except Exception as e:
                if debug:
                    print(f"  Error processing volume {volume_id}: {e}")
                continue
        
        # Create DataFrame
        if all_papers:
            df = pd.DataFrame(all_papers)
        else:
            df = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'year', 'authors', 
                                      'venue', 'url', 'citations', 'conference', 
                                      'classification_types', 'domains', "model_types", 'benchmarks', 
                                      'num_benchmarks', 'performance'])
        
        print(f"\n{'='*60}")
        print(f"Debug stats:")
        print(f"  Total volumes checked: {volumes_checked}")
        print(f"  Years found in range: {sorted(years_found) if years_found else 'None'}")
        print(f"  Volumes matched (year + conf): {volumes_matched}")
        print(f"  Total papers checked: {papers_checked}")
        print(f"  Classification papers found: {len(all_papers)}")
        print(f"{'='*60}\n")
        
        return df
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate cartography statistics"""
        if len(df) == 0:
            return {
                'total_papers': 0,
                'papers_by_year': {},
                'papers_by_conference': {},
                'classification_types_dist': {},
                'domains_dist': {},
                'avg_benchmarks': 0,
                'total_benchmarks': 0,
                'highly_cited': 0,
                'low_cited': 0,
            }
        
        stats = {
            'total_papers': len(df),
            'papers_by_year': df['year'].value_counts().sort_index().to_dict(),
            'papers_by_conference': df['conference'].value_counts().to_dict(),
            'classification_types_dist': defaultdict(int),
            'domains_dist': defaultdict(int),
            'models_dist': defaultdict(int),
            'avg_benchmarks': df['num_benchmarks'].mean(),
            'total_benchmarks': int(df['num_benchmarks'].sum()),
            'highly_cited': int(len(df[df['citations'] > 200])),
            'low_cited': int(len(df[df['citations'] <= 200])),
        }
        
        # Type distribution
        for types in df['classification_types']:
            for t in types:
                stats['classification_types_dist'][t] += 1
                
        # Domain distribution
        for domains in df['domains']:
            for d in domains:
                stats['domains_dist'][d] += 1

        for model_types in df['model_types']:
            for m in model_types:
                stats['models_dist'][m] += 1
                
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict, filename: str = 'acl_cartography'):
        """Save results to files"""
        # Save DataFrame
        df.to_csv(f'{filename}.csv', index=False, encoding='utf-8')
        df.to_json(f'{filename}.json', orient='records', indent=2, force_ascii=False)
        
        # Save statistics
        import json
        with open(f'{filename}_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Save human-readable summary
        with open(f'{filename}_summary.txt', 'w', encoding='utf-8') as f:
            f.write("ACL ANTHOLOGY CLASSIFICATION CARTOGRAPHY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total classification papers: {stats['total_papers']}\n\n")
            
            if stats['total_papers'] > 0:
                f.write("Papers by year:\n")
                for year, count in sorted(stats['papers_by_year'].items()):
                    f.write(f"  {year}: {count}\n")
                
                f.write("\nPapers by conference:\n")
                for conf, count in sorted(stats['papers_by_conference'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    f.write(f"  {conf.upper()}: {count}\n")
                
                f.write("\nClassification types:\n")
                for type_name, count in sorted(stats['classification_types_dist'].items(),
                                              key=lambda x: x[1], reverse=True):
                    f.write(f"  {type_name}: {count}\n")
                
                f.write("\nDomains:\n")
                for domain, count in sorted(stats['domains_dist'].items(),
                                           key=lambda x: x[1], reverse=True):
                    f.write(f"  {domain}: {count}\n")

                # --- NEW SECTION: Top Models/Architectures ---
                f.write("\nTop Models/Architectures:\n")
                model_dist = stats.get('models_dist', {})
                if model_dist:
                    for model_name, count in sorted(model_dist.items(), 
                                                    key=lambda x: x[1], reverse=True):
                        f.write(f"  {model_name}: {count}\n")
                else:
                    f.write("  (No model distribution data available)\n")
                # --------------------------------------------
                
                f.write(f"\nBenchmarks:\n")
                f.write(f"  Average per paper: {stats['avg_benchmarks']:.2f}\n")
                f.write(f"  Total mentioned: {stats['total_benchmarks']}\n")
                
                f.write(f"\nCitations:\n")
                f.write(f"  Highly cited (>200): {stats['highly_cited']}\n")
                f.write(f"  Low cited (≤200): {stats['low_cited']}\n")
        
        print(f"\nResults saved:")
        print(f"  - {filename}.csv (full data)")
        print(f"  - {filename}.json (full data)")
        print(f"  - {filename}_stats.json (statistics)")
        print(f"  - {filename}_summary.txt (human-readable summary)")


def main():
    """Main function"""
    
    print("="*60)
    print("ACL ANTHOLOGY CLASSIFICATION CARTOGRAPHY")
    print("IFT6285 - Project 1")
    print("="*60)
    
    # Initialize cartographer (2010-2025)
    cartographer = ACLCartographer(start_year=1965, end_year=2025)
    
    # Run cartography with debug mode
    df = cartographer.run_cartography(debug=True)
    
    # Generate statistics
    stats = cartographer.generate_statistics(df)
    
    # Display summary
    print("\n" + "="*60)
    print("CARTOGRAPHY SUMMARY")
    print("="*60)
    print(f"Total classification papers: {stats['total_papers']}")
    
    if stats['total_papers'] > 0:
        print(f"\nPapers by year:")
        for year, count in sorted(stats['papers_by_year'].items()):
            print(f"  {year}: {count}")
        print(f"\nPapers by conference:")
        for conf, count in sorted(stats['papers_by_conference'].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"  {conf.upper()}: {count}")
        print(f"\nClassification types:")
        for type_name, count in sorted(stats['classification_types_dist'].items(),
                                      key=lambda x: x[1], reverse=True):
            print(f"  {type_name}: {count}")
        print(f"\nDomains:")
        for domain, count in sorted(stats['domains_dist'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count}")
        print(f"\nBenchmarks:")
        print(f"  Average per paper: {stats['avg_benchmarks']:.2f}")
        print(f"  Total mentioned: {stats['total_benchmarks']}")
        print(f"\nCitations:")
        print(f"  Highly cited (>200): {stats['highly_cited']}")
        print(f"  Low cited (≤200): {stats['low_cited']}")
        
        # Save results
        cartographer.save_results(df, stats)
        
    else:
        print("\nNo classification papers found.")
    
    print("\n" + "="*60)
    print("Cartography complete!")
    print("="*60)


if __name__ == "__main__":
    main()