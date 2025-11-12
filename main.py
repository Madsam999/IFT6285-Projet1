from cartography import ACLCartographer

def main():
    print("hello world")

        # Initialize cartographer (2020-2025)
    cartographer = ACLCartographer(1965, 2025)
    
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
            print(f"  Title: {row.get('title', 'N/A')}")
            print(f"  Year: {row['year']} | Conf: {row['conference'].upper()} | Citations: {row['citations']}")
            print(f"  Types: {', '.join(row['classification_types'])}")
            print(f"  Domains: {', '.join(row['domains'])}")
            if row['benchmarks']:
                print(f"  Benchmarks: {', '.join(row['benchmarks'][:5])}")
    else:
        print("\nNo classification papers found.")
    
    print("\n" + "="*60)
    print("Cartography complete!")
    print("="*60)



if __name__ == "__main__":
    main()
