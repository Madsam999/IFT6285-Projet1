"""
ACL Anthology Classification Paper Cartography Script

This script collects and analyzes classification papers from
ACL, NAACL, EACL, CoNLL, EMNLP, COLING and LREC conferences
"""

from acl_anthology import Anthology
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
import re
from datetime import datetime


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
        'lrec': ['lrec']
    }
    
    # Keywords for identifying classification types
    CLASSIFICATION_KEYWORDS = [
        'classification', 'classifier', 'categorization', 'labeling',
        'sentiment analysis', 'sentiment classification', 
        'text classification', 'document classification',
        'binary classification', 'multi-class', 'multi-label',
        'benchmark', 'dataset', 'corpus'
    ]
    
    # Classification types
    CLASSIFICATION_TYPES = {
        'binary': ['binary classification', 'binary', 'two-class'],
        'multi-class': ['multi-class', 'multiclass', 'multiple classes'],
        'multi-label': ['multi-label', 'multilabel', 'multiple labels']
    }
    
    # Classification domains
    DOMAINS = {
        'sentiment': ['sentiment', 'opinion', 'polarity', 'emotion'],
        'topic': ['topic', 'subject', 'category'],
        'intent': ['intent', 'intention'],
        'stance': ['stance detection'],
        'toxicity': ['toxic', 'hate speech', 'offensive'],
        'fake_news': ['fake news', 'misinformation', 'fact'],
        'ner': ['named entity', 'ner'],
        'other': []
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
        
    def _extract_paper_metadata(self, paper) -> Dict:
        """
        Extract metadata from a paper object
        
        Args:
            paper: Paper object from acl-anthology
            
        Returns:
            Dictionary with paper metadata
        """
        try:
            # Extract basic info
            metadata = {
                'paper_id': str(paper.id),
                'title': str(paper.title),
                'abstract': str(paper.abstract) if paper.abstract else "",
                'year': paper.year,
                'authors': [str(author.full) if hasattr(author, 'full') else str(author) for author in paper.authors] if paper.authors else [],
                'venue': str(paper.parent_volume.title) if hasattr(paper, 'parent_volume') and paper.parent_volume else "",
                'url': f"https://aclanthology.org/{paper.id}/",
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
        for keyword in self.CLASSIFICATION_KEYWORDS:
            if keyword.lower() in text:
                return True
                
        return False
    
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
            for keyword in keywords:
                if keyword.lower() in text:
                    types.append(type_name)
                    break
                    
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
            for keyword in keywords:
                if keyword.lower() in text:
                    domains.append(domain_name)
                    break
                    
        return domains if domains else ['other']
    
    def extract_benchmarks(self, paper: Dict) -> List[str]:
        """
        Extract mentioned benchmark names
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            List of benchmarks
        """
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".upper()
        benchmarks = []
        
        # Common benchmark patterns
        benchmark_patterns = [
            r'\b([A-Z]{3,}(?:-[A-Z0-9]+)?)\b',  # E.g., GLUE, SST-2
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # E.g., SemEval, CoNLL
        ]
        
        for pattern in benchmark_patterns:
            matches = re.findall(pattern, text)
            benchmarks.extend(matches)
            
        # Filter common false positives
        false_positives = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'ACL', 'NLP', 'AI', 
                          'NER', 'POS', 'QA', 'MT', 'NLU', 'NLG', 'USA', 'UK'}
        benchmarks = [b for b in benchmarks if b not in false_positives]
        
        return list(set(benchmarks))[:10]  # Limit to 10 benchmarks
    
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
        
        # Common metric patterns
        metric_patterns = {
            'accuracy': r'accuracy[:\s]+(\d+\.?\d*)%?',
            'f1': r'f1[:\s-]+(\d+\.?\d*)%?',
            'precision': r'precision[:\s]+(\d+\.?\d*)%?',
            'recall': r'recall[:\s]+(\d+\.?\d*)%?',
        }
        
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, abstract)
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
        
        all_papers = []
        volumes_checked = 0
        volumes_matched = 0
        papers_checked = 0
        
        # Iterate through all volumes in the anthology
        for volume in self.anthology.volumes():
            volumes_checked += 1
            volume_id = str(volume.full_id)
            
            # Check if this volume matches our criteria
            try:
                # Extract year from volume ID (format: YYYY.conference-track)
                year_match = re.match(r'(\d{4})\.([a-z]+)', volume_id)
                if not year_match:
                    continue
                    
                year = int(year_match.group(1))
                conf_prefix = year_match.group(2)
                
                # Check if year is in range
                if year < self.start_year or year > self.end_year:
                    continue
                
                # Check if conference is in our list
                conf_name = None
                for conf_key, conf_prefixes in self.CONFERENCES.items():
                    if conf_prefix in conf_prefixes:
                        conf_name = conf_key
                        break
                
                if not conf_name:
                    continue
                
                volumes_matched += 1
                if debug and volumes_matched <= 3:
                    print(f"  Processing volume: {volume_id}")
                
                # Process papers in this volume
                volume_papers = 0
                for paper in volume.papers():
                    papers_checked += 1
                    paper_data = self._extract_paper_metadata(paper)
                    
                    if debug and volumes_matched <= 3 and volume_papers < 2:
                        print(f"    Paper: {paper.id}")
                        print(f"      Title: {paper_data.get('title', 'N/A')[:60] if paper_data else 'ERROR'}")
                        print(f"      Has abstract: {bool(paper_data.get('abstract')) if paper_data else False}")
                    
                    if paper_data and self.is_classification_paper(paper_data):
                        paper_data['conference'] = conf_name
                        paper_data['classification_types'] = self.classify_paper_type(paper_data)
                        paper_data['domains'] = self.classify_domain(paper_data)
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
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'year', 'authors', 
                                      'venue', 'url', 'citations', 'conference', 
                                      'classification_types', 'domains', 'benchmarks', 
                                      'num_benchmarks', 'performance'])
        
        print(f"\n{'='*60}")
        print(f"Debug stats:")
        print(f"  Total volumes checked: {volumes_checked}")
        print(f"  Volumes matched (year + conf): {volumes_matched}")
        print(f"  Total papers checked: {papers_checked}")
        print(f"  Classification papers found: {len(all_papers)}")
        print(f"{'='*60}\n")
        
        return df
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate cartography statistics
        
        Args:
            df: Papers DataFrame
            
        Returns:
            Statistics dictionary
        """
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
            'avg_benchmarks': df['num_benchmarks'].mean(),
            'total_benchmarks': df['num_benchmarks'].sum(),
            'highly_cited': len(df[df['citations'] > 200]),
            'low_cited': len(df[df['citations'] <= 200]),
        }
        
        # Type distribution
        for types in df['classification_types']:
            for t in types:
                stats['classification_types_dist'][t] += 1
                
        # Domain distribution
        for domains in df['domains']:
            for d in domains:
                stats['domains_dist'][d] += 1
                
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict, filename: str = 'acl_cartography'):
        """
        Save results to files
        
        Args:
            df: Papers DataFrame
            stats: Statistics dictionary
            filename: Base filename for output
        """
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
    
    # Initialize cartographer (2020-2025)
    cartographer = ACLCartographer(start_year=2020, end_year=2025)
    
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
        
        # Display top under-explored papers (low citations, good candidates for study)
        print("\n" + "="*60)
        print("TOP 15 UNDER-EXPLORED PAPERS (study candidates)")
        print("Papers with ≤200 citations, sorted by most recent")
        print("="*60)
        
        low_cited = df[df['citations'] <= 200].sort_values(['year', 'citations'], 
                                                            ascending=[False, False]).head(15)
        
        for idx, row in low_cited.iterrows():
            print(f"\n[{row['paper_id']}]")
            print(f"  Title: {row.get('title', 'N/A')[:75]}")
            print(f"  Year: {row['year']} | Conf: {row['conference'].upper()} | Citations: {row['citations']}")
            print(f"  Types: {', '.join(row['classification_types'])}")
            print(f"  Domains: {', '.join(row['domains'])}")
            if row['benchmarks']:
                print(f"  Benchmarks: {', '.join(row['benchmarks'][:5])}")
            print(f"  URL: {row['url']}")
    else:
        print("\nNo classification papers found.")
    
    print("\n" + "="*60)
    print("Cartography complete!")
    print("="*60)


if __name__ == "__main__":
    main()